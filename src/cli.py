"""Nova Trader CLI.

Three verbs:
    nova run --tickers AAPL,NVDA      Live run — fetches data, calls agents, records to disk.
    nova show <run_id>                View-only — re-prints a saved Recommendation.
    nova rerun <run_id>               Re-execute against the saved snapshot. Saves a new run.

Every run writes to ~/.nova-trader/runs/<run_id>/ (overridable via NOVA_RUNS_DIR env var).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.table import Table

from src.chat_cli import launch_chat
from src.runs import RunRecorder, runs_root
from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio, Position, RealizedGains
from src.schemas.signals import Recommendation
from src.schemas.snapshot import MarketSnapshot
from src.utils.progress import progress
from src.engine import run_engine
from src.registry import AGENT_REGISTRY, all_agent_ids

# ── Theme: warm earth tones (matches the legacy nova CLI) ───
INK = "grey85"
STONE = "grey62"
ASH = "grey50"
DARK_ASH = "grey35"
SAGE = "#a7c080"     # bullish / buy
TEAL = "#7fbbb3"     # headers
ROSE = "#e67e80"     # bearish / sell / short
AMBER = "#dbbc7f"    # neutral / hold

SYM_DIAMOND = "◆"

console = Console()


def _default_start_date() -> str:
    return (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d")


def _default_end_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nova", description="Nova Trader — recommendation engine")
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    # ── nova run ──
    run = sub.add_parser("run", help="Live run: fetch data, run agents, record to disk")
    run.add_argument("--tickers", required=True, help="Comma-separated, e.g. AAPL,NVDA")
    run.add_argument(
        "--start-date",
        default=_default_start_date(),
        help="YYYY-MM-DD (default: 3 months ago)",
    )
    run.add_argument(
        "--end-date",
        default=_default_end_date(),
        help="YYYY-MM-DD (default: today)",
    )
    run.add_argument("--initial-cash", type=float, default=100_000.0)
    run.add_argument("--margin-requirement", type=float, default=0.5)
    run.add_argument(
        "--agents", default="",
        help=f"Comma-separated agent ids (default: all). Available: {','.join(all_agent_ids())}",
    )
    run.add_argument("--model-name", default=os.getenv("NOVA_MODEL_NAME", "MiniMax-M2.7"))
    run.add_argument("--model-provider", default=os.getenv("NOVA_MODEL_PROVIDER", "MiniMax"))
    run.add_argument(
        "--portfolio-mode",
        choices=["research", "long_only", "long_short"],
        default=os.getenv("NOVA_PORTFOLIO_MODE", "research"),
        help="research shows direct recommendations; long_short requires a short hedge for opening buys",
    )
    run.add_argument("--seed", type=int, default=None, help="Override the LLM seed (default: derived from run_id)")
    run.add_argument("--json", action="store_true", help="Print the full Recommendation as JSON")
    run.add_argument("--no-progress", action="store_true")
    run.add_argument("--no-record", action="store_true", help="Don't write to ~/.nova-trader/runs/")

    # ── nova show <run_id> ──
    show = sub.add_parser("show", help="View a saved run's Recommendation (no re-execution)")
    show.add_argument("run_id", help="Run id (the directory name under ~/.nova-trader/runs/)")
    show.add_argument("--json", action="store_true")

    # ── nova rerun <run_id> ──
    rerun = sub.add_parser("rerun", help="Replay a saved run against its cached snapshot")
    rerun.add_argument("run_id", help="Run id to replay")
    rerun.add_argument("--json", action="store_true")
    rerun.add_argument("--no-progress", action="store_true")

    # ── nova debate <ticker> <question> ──
    debate = sub.add_parser("debate", help="Research-desk debate: supervisor + 4 specialists + cited memo")
    debate.add_argument("ticker", help="Ticker to research, e.g. NVDA")
    debate.add_argument("question", nargs="+", help="The research question")
    debate.add_argument("--horizon", default="6-12 months")
    debate.add_argument("--source", choices=["filings", "web"], default="filings")
    debate.add_argument("--fast", action="store_true", help="Haiku specialists + Opus bear (~5 min)")
    debate.add_argument("--quick", action="store_true", help="Haiku specialists + Sonnet bear, no web (quickest, ~3-4 min)")

    # ── nova web ──
    web = sub.add_parser("web", help="Start the browser demo UI")
    web.add_argument("--host", default=os.getenv("NOVA_WEB_HOST", "127.0.0.1"))
    web.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    web.add_argument("--model-name", default=os.getenv("NOVA_MODEL_NAME", "claude-opus-4-8"))
    web.add_argument("--model-provider", default=os.getenv("NOVA_MODEL_PROVIDER", "Anthropic"))
    web.add_argument("--reload", action="store_true", help="Reload the web server on code changes")

    return p


def _build_context_for_run(args: argparse.Namespace) -> RunContext:
    """Build a fresh RunContext for `nova run`."""
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("--tickers must contain at least one symbol")

    portfolio = Portfolio(
        cash=args.initial_cash,
        margin_requirement=args.margin_requirement,
        margin_used=0.0,
        positions={t: Position() for t in tickers},
        realized_gains={t: RealizedGains() for t in tickers},
    )
    request = RunRequest(
        tickers=tickers,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        portfolio=portfolio,
        model=ModelConfig(provider=args.model_provider, name=args.model_name),
        portfolio_mode=args.portfolio_mode,
        selected_agents=[a.strip() for a in args.agents.split(",") if a.strip()],
    )
    return RunContext(
        request=request,
        as_of=datetime.now(timezone.utc),
        seed=args.seed,
    )


def _build_context_from_metadata(meta: dict) -> tuple[RunContext, MarketSnapshot]:
    """Rebuild a RunContext + load the cached snapshot for a `nova rerun`.

    The rerun gets a NEW run_id (so it doesn't overwrite the original record),
    but reuses the original seed + tickers + dates so the LLM result should
    match if nothing else changed.
    """
    req_data = {
        "tickers": meta["tickers"],
        "start_date": date.fromisoformat(meta["start_date"]),
        "end_date": date.fromisoformat(meta["end_date"]),
        "portfolio": Portfolio(
            cash=100_000.0,  # Portfolio cash is irrelevant for replay analytics.
            margin_requirement=0.5,
            positions={t: Position() for t in meta["tickers"]},
            realized_gains={t: RealizedGains() for t in meta["tickers"]},
        ),
        "model": ModelConfig(**meta["model"]),
        "portfolio_mode": meta.get("portfolio_mode", "research"),
        "selected_agents": meta.get("selected_agents", []),
    }
    request = RunRequest(**req_data)
    ctx = RunContext(
        request=request,
        as_of=datetime.now(timezone.utc),
        seed=meta.get("seed"),
    )
    snapshot_data = RunRecorder.load_snapshot_dict(meta["run_id"])
    snapshot = MarketSnapshot.model_validate(snapshot_data)
    return ctx, snapshot


def _action_color(action: str) -> str:
    a = action.upper()
    if a in ("BUY", "STRONG_BUY", "COVER"):
        return SAGE
    if a in ("SELL", "STRONG_SELL", "SHORT"):
        return ROSE
    return AMBER


def _direction_color(direction: str) -> str:
    d = direction.upper()
    if d == "BULLISH":
        return SAGE
    if d == "BEARISH":
        return ROSE
    return AMBER


def _display_name(agent_id: str) -> str:
    spec = AGENT_REGISTRY.get(agent_id)
    if spec:
        return spec.display_name
    return agent_id.replace("_", " ").title()


def _print_human(recommendation) -> None:
    console.print()
    console.print(
        f"  {SYM_DIAMOND} Run [{ASH}]{recommendation.run_id}[/{ASH}]",
        style=f"bold {TEAL}",
    )
    console.print()

    # Group signals per ticker once for easy access.
    signals_by_ticker: dict[str, list] = {}
    for s in recommendation.signals:
        signals_by_ticker.setdefault(s.ticker, []).append(s)

    for ticker in recommendation.tickers:
        decision = recommendation.decisions.per_ticker.get(ticker)
        if decision is None:
            continue

        action = decision.action.upper()
        confidence = decision.confidence
        action_color = _action_color(action)

        # Header: TICKER → ACTION qty shares (xx%)
        console.print(f"  {ticker}", style=f"bold {TEAL}", end="")
        console.print("  →  ", style=ASH, end="")
        console.print(action, style=f"bold {action_color}", end="")
        if decision.quantity:
            console.print(f" {decision.quantity} shares", style=STONE, end="")
        console.print(f"  ({confidence:.0%})", style=ASH)
        console.print()

        # Explicit column widths so columns don't grow to fill wide terminals.
        # All three tables share the first-column width for vertical alignment.
        col1_width = 32        # Analyst / Why / Risk all use this (fits "● News Sentiment Analyst")
        col2_width = 10        # Signal column width inside analysts table
        wrap_width = 56        # Reasoning / Value columns wrap inside this width

        # Analysts table — clean columns, no reasoning column (kept short)
        analysts_table = Table(
            box=box.SIMPLE,
            header_style=f"bold {TEAL}",
            show_edge=False,
            pad_edge=False,
            padding=(0, 2),
        )
        analysts_table.add_column("Analyst", style=STONE, no_wrap=True, width=col1_width)
        analysts_table.add_column("Signal", no_wrap=True, width=col2_width)
        analysts_table.add_column("Confidence", no_wrap=True, width=12)
        analysts_table.add_column("%", justify="right", no_wrap=True, width=6)

        reasonings: list[tuple[str, str, str]] = []  # (name, color, text)

        for sig in signals_by_ticker.get(ticker, []):
            name = _display_name(sig.agent_id)
            if sig.status == "failed":
                direction, color = "FAILED", ROSE
                bar, pct = "—", "—"
                reasonings.append((name, ROSE, sig.error or "agent crashed"))
            elif sig.status == "abstained":
                direction, color = "ABSTAIN", ASH
                bar, pct = "—", "—"
                reasonings.append((name, ASH, sig.reasoning or "no opinion"))
            else:
                direction = sig.direction.upper()
                color = _direction_color(direction)
                filled = int(sig.confidence * 10)
                bar = "█" * filled + "░" * (10 - filled)
                pct = f"{sig.confidence:.0%}"
                if sig.reasoning:
                    reasonings.append((name, color, sig.reasoning))

            analysts_table.add_row(
                name,
                f"[bold {color}]{direction}[/bold {color}]",
                f"[{color}]{bar}[/{color}]",
                pct,
            )

        console.print(Padding(analysts_table, (0, 0, 1, 2)))

        # Reasoning — its own table per ticker so text wraps inside the column
        if reasonings:
            why_table = Table(
                box=box.SIMPLE,
                header_style=f"bold {TEAL}",
                show_edge=False,
                pad_edge=False,
                padding=(0, 2),
            )
            why_table.add_column("Why", style=STONE, no_wrap=True, width=col1_width)
            why_table.add_column("Reasoning", style=INK, overflow="fold", width=wrap_width)
            for name, color, text in reasonings:
                why_table.add_row(
                    f"[{color}]●[/{color}] {name}",
                    text,
                )
            console.print(Padding(why_table, (0, 0, 1, 2)))

        # Risk + Decision — single table so the Decision value stays in-column
        lim = recommendation.limits.per_ticker.get(ticker)
        risk_table = Table(
            box=box.SIMPLE,
            header_style=f"bold {TEAL}",
            show_edge=False,
            pad_edge=False,
            padding=(0, 2),
        )
        risk_table.add_column("Risk", style=ASH, no_wrap=True, width=col1_width)
        risk_table.add_column("Value", style=INK, overflow="fold", width=wrap_width)
        if lim:
            risk_table.add_row("Price",        f"${lim.current_price:,.2f}")
            risk_table.add_row("Volatility",   f"{lim.annualized_volatility:.1%} annualized")
            risk_table.add_row("Max position", f"{lim.max_shares} shares  (${lim.max_position_dollars:,.0f})")
            risk_table.add_row("Correlation",  f"×{lim.correlation_multiplier:.2f}")
        # Decision as a row inside the same table, styled to stand out.
        decision_text = decision.reasoning or "—"
        if decision.hedge_pair_id:
            decision_text += f"  [{ASH}](hedge pair: {decision.hedge_pair_id})[/{ASH}]"
        risk_table.add_row(
            f"[bold {TEAL}]Decision[/bold {TEAL}]",
            decision_text,
        )
        console.print(Padding(risk_table, (0, 0, 1, 2)))

    # Hedge plan summary
    plan = recommendation.decisions.hedge_plan
    if plan.pairs or plan.blocked_longs:
        console.print(f"  {SYM_DIAMOND} Hedge plan  [{ASH}]status={plan.status}[/{ASH}]", style=f"bold {TEAL}")
        for p in plan.pairs:
            console.print(
                f"    LONG {p.long_quantity} [bold {SAGE}]{p.long_ticker}[/bold {SAGE}]  "
                f"vs  SHORT {p.short_quantity} [bold {ROSE}]{p.short_ticker}[/bold {ROSE}]  "
                f"[{ASH}]ratio {p.hedge_ratio:.2f}[/{ASH}]",
                style=STONE,
            )
        if plan.blocked_longs:
            console.print(
                f"    [bold {AMBER}]blocked[/bold {AMBER}] (no hedge): "
                f"{', '.join(plan.blocked_longs)}",
                style=STONE,
            )
        console.print()

    # Signal health footer
    ok = sum(1 for s in recommendation.signals if s.status == "ok")
    abstained = sum(1 for s in recommendation.signals if s.status == "abstained")
    failed = sum(1 for s in recommendation.signals if s.status == "failed")
    console.print(
        f"  [{ASH}]{len(recommendation.signals)} signals: "
        f"[{SAGE}]{ok} ok[/{SAGE}]  "
        f"[{AMBER}]{abstained} abstained[/{AMBER}]  "
        f"[{ROSE}]{failed} failed[/{ROSE}][/{ASH}]"
    )
    console.print()


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        ctx = _build_context_for_run(args)
    except Exception as e:
        print(f"Bad arguments: {e}", file=sys.stderr)
        return 2

    if not args.no_progress:
        progress.start()
    try:
        recommendation = run_engine(
            ctx,
            selected_agents=ctx.request.selected_agents or None,
            record=not args.no_record,
        )
    finally:
        if not args.no_progress:
            progress.stop()

    if args.json:
        print(recommendation.model_dump_json(indent=2))
    else:
        _print_human(recommendation)
        if not args.no_record:
            console.print(
                f"  [{ASH}]saved to {runs_root() / ctx.run_id}/[/{ASH}]"
            )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    if not RunRecorder.exists(args.run_id):
        print(f"No run found at {runs_root() / args.run_id}", file=sys.stderr)
        return 2
    try:
        recommendation = RunRecorder.load_recommendation(args.run_id)
    except FileNotFoundError:
        print(f"Run {args.run_id} has no recommendation.json (incomplete run?)", file=sys.stderr)
        return 2
    if args.json:
        print(recommendation.model_dump_json(indent=2))
    else:
        _print_human(recommendation)
    return 0


def _cmd_rerun(args: argparse.Namespace) -> int:
    if not RunRecorder.exists(args.run_id):
        print(f"No run found at {runs_root() / args.run_id}", file=sys.stderr)
        return 2
    meta = RunRecorder.load_metadata(args.run_id)
    ctx, snapshot = _build_context_from_metadata(meta)

    if not args.no_progress:
        progress.start()
    try:
        recommendation = run_engine(
            ctx,
            selected_agents=ctx.request.selected_agents or None,
            snapshot=snapshot,
            record=True,
        )
    finally:
        if not args.no_progress:
            progress.stop()

    if args.json:
        print(recommendation.model_dump_json(indent=2))
    else:
        _print_human(recommendation)
        console.print(
            f"  [{ASH}]rerun saved to {runs_root() / ctx.run_id}/  "
            f"(original: {args.run_id})[/{ASH}]"
        )
    return 0


def _cmd_web(args: argparse.Namespace) -> int:
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["MODEL_PROVIDER"] = args.model_provider
    from src.web.server import launch_web

    return launch_web(args.host, args.port, reload=args.reload)


def _cmd_debate(args: argparse.Namespace) -> int:
    """One-shot research-desk debate printed to the terminal (additive; does not
    touch the deterministic analyst pipeline)."""
    import asyncio

    from dotenv import load_dotenv as _ld

    _ld(override=True)  # .env authoritative (avoid a stale shell ANTHROPIC_API_KEY)
    from src.debate import USAGE, run_debate
    from src.debate.recorder import DebateRecorder

    ticker = args.ticker.upper()
    question = " ".join(args.question)
    recorder = DebateRecorder(ticker, question, args.horizon, args.source)

    def emit(**ev) -> None:
        recorder.event(**ev)
        t = ev.get("type")
        if t == "phase":
            print(f"[{ev.get('phase')}] {ev.get('status')} {ev.get('detail', '')}", flush=True)
        elif t == "tool_call":
            bits = ev.get("query") or (", ".join(ev.get("results", [])[:3]) if ev.get("results") else "")
            print(f"    {ev.get('agent', '')} {ev.get('tool', '')}: {bits}", flush=True)
        elif t == "specialist":
            extra = ""
            if ev.get("status") == "done":
                d = ev.get("draft") or {}
                extra = f"-> {d.get('stance')} ({len(d.get('key_findings', []))} findings)"
            print(f"  specialist {ev.get('key')}: {ev.get('status')} {extra}", flush=True)

    async def driver():
        return await run_debate(ticker, question, args.horizon, args.source,
                                emit=emit, record=recorder.record,
                                fast=args.fast or args.quick,
                                bear_model="claude-sonnet-4-6" if args.quick else None,
                                bear_web=not args.quick)

    try:
        result, _store = asyncio.run(driver())
    except Exception as exc:  # noqa: BLE001
        print(f"Debate failed: {type(exc).__name__}: {exc}")
        return 1

    run_id = recorder.save(result, str(USAGE))

    memo = result.get("memo", {})
    bar = "=" * 70
    print(f"\n{bar}\n RESEARCH MEMO — {memo.get('ticker', ticker)}  "
          f"({result.get('specialist_source', '?')})\n{bar}")
    print(f" Conviction: {str(memo.get('conviction', '')).upper()}  |  "
          f"Lean: {str(memo.get('directional_lean', '')).upper()}")
    print(f"\n Bull: {memo.get('bull_case', '')}")
    print(f" Base: {memo.get('base_case', '')}")
    print(f" Bear: {memo.get('bear_case', '')}")
    if memo.get("key_risks"):
        print("\n Key risks:")
        for r in memo["key_risks"]:
            print(f"   • {r}")
    cites = memo.get("citations", [])
    if cites:
        print(f"\n Citations ({len(cites)}): " + ", ".join(cites[:12]))
    print(f"\n usage: {USAGE}")
    print(f" saved trajectory: {runs_root() / run_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    raw_argv = sys.argv[1:] if argv is None else argv
    if not raw_argv:
        if sys.stdin.isatty():
            return launch_chat(
                console,
                provider=os.getenv("NOVA_MODEL_PROVIDER", "MiniMax"),
                model=os.getenv("NOVA_MODEL_NAME", "MiniMax-M2.7"),
                portfolio_mode=os.getenv("NOVA_PORTFOLIO_MODE", "research"),
            )
        parser = _build_arg_parser()
        parser.print_help()
        return 2

    parser = _build_arg_parser()
    args = parser.parse_args(raw_argv)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "show":
        return _cmd_show(args)
    if args.command == "rerun":
        return _cmd_rerun(args)
    if args.command == "web":
        return _cmd_web(args)
    if args.command == "debate":
        return _cmd_debate(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
