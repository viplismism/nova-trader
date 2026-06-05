"""Rendering and text helpers for Nova chat."""

from __future__ import annotations

import re

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.chat.theme import AMBER, ASH, INK, ROSE, SAGE, TEAL
from src.runs import runs_root
from src.schemas.signals import Recommendation


STATUS_PHRASES = {
    "calling llm": "reasoning with the model",
    "deciding": "making the final decision",
    "running buffett scoring": "scoring quality & value",
}


def humanize_status(status: str) -> str:
    return STATUS_PHRASES.get(status.strip().lower(), status)


def clean_action(status: str) -> str:
    """A short, number-free verb describing what an agent is doing."""
    low = status.strip().lower()
    if low in STATUS_PHRASES:
        return STATUS_PHRASES[low]
    if "fetch" in low:
        return "gathering data"
    if "limit" in low or "portfolio value" in low or "margin" in low:
        return "sizing positions"
    if "classif" in low or "headline" in low:
        return "reading the news"
    if "insider" in low:
        return "weighing insider activity"
    if any(k in low for k in ("scor", "comput", "analyz", "running", "trend", "valuation", "growth")):
        return "analyzing"
    if "calling llm" in low:
        return "reasoning with the model"
    return "working"


def status_color(value: str) -> str:
    normalized = value.lower()
    if normalized in {"buy", "bullish", "ok"}:
        return SAGE
    if normalized in {"sell", "short", "bearish", "failed"}:
        return ROSE
    return AMBER


def field_value(obj: object, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def stream_markdown_display_text(text: str) -> str:
    """Return markdown that is safe to render while tokens are still arriving."""
    cleaned = text

    if cleaned.count("**") % 2:
        idx = cleaned.rfind("**")
        if idx != -1:
            cleaned = cleaned[:idx] + cleaned[idx + 2:]

    cleaned = re.sub(r"(?<!\*)\*(?!\*)", "", cleaned)

    if cleaned.count("`") % 2:
        idx = cleaned.rfind("`")
        if idx != -1:
            cleaned = cleaned[:idx] + cleaned[idx + 1:]

    return cleaned


SIMPLE_QUESTIONS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "what is this",
    "what is this?",
    "what's this",
    "what's this?",
    "what's this is",
    "what's this is?",
    "whats this",
    "whats this?",
    "who are you",
    "who are you?",
    "help",
    "/help",
    "?",
}


def simple_response(text: str) -> str | None:
    normalized = " ".join(text.strip().lower().split())
    if normalized in {"hi", "hello", "hey", "yo"}:
        return (
            "Hey. I’m Nova Trader, a portfolio-aware equity research agent.\n\n"
            "Ask `analyze AAPL,NVDA` to run the analyst pipeline, or ask a finance question."
        )
    if normalized in {"help", "/help", "?"}:
        return None
    if normalized in SIMPLE_QUESTIONS or normalized.startswith(("what is this", "what's this", "whats this", "who are you")):
        return (
            "Nova Trader is a portfolio-aware equity research workspace.\n\n"
            "- It runs multiple analysts across tickers.\n"
            "- It combines signals into a consensus.\n"
            "- It applies risk limits and explicit portfolio-mode rules.\n"
            "- It keeps the run auditable so you can inspect each analyst's reasoning.\n\n"
            "Try `analyze AAPL,NVDA`, then `details AAPL`."
        )
    return None


def recommendation_summary_text(recommendation: Recommendation) -> str:
    lines = [
        f"Run {recommendation.run_id}",
        f"Summary: {recommendation.summary}",
        "",
        f"{'Ticker':<8} {'Consensus':<10} {'Action':<8} {'Conf':>5}  Decision",
        "-" * 74,
    ]

    for ticker in recommendation.tickers:
        consensus = recommendation.consensus.get(ticker)
        decision = recommendation.decisions.per_ticker.get(ticker)
        if not decision:
            continue
        consensus_text = str(field_value(consensus, "direction", "n/a")).upper()
        action = str(field_value(decision, "action", "hold")).upper()
        confidence = float(field_value(decision, "confidence", 0.0))
        reasoning = str(field_value(decision, "reasoning", "") or "")
        lines.append(f"{ticker:<8} {consensus_text:<10} {action:<8} {confidence:>4.0%}  {reasoning}")

    plan = recommendation.decisions.hedge_plan
    if plan.pairs or plan.blocked_longs:
        lines.extend(["", f"Hedge plan: {plan.status}"])
        for pair in plan.pairs:
            lines.append(
                f"  LONG {pair.long_quantity} {pair.long_ticker} vs "
                f"SHORT {pair.short_quantity} {pair.short_ticker}"
            )
        if plan.blocked_longs:
            lines.append(f"  Blocked longs: {', '.join(plan.blocked_longs)}")

    lines.append("")
    lines.append("Analyst reasoning")
    lines.append("-" * 74)
    for ticker in recommendation.tickers:
        lines.append(f"{ticker}")
        ticker_signals = [signal for signal in recommendation.signals if signal.ticker == ticker]
        if not ticker_signals:
            lines.append("  No analyst signals.")
            continue
        for signal in ticker_signals:
            status = signal.status
            direction = signal.direction.upper()
            confidence = f"{signal.confidence:.0%}" if status == "ok" else status
            reasoning = signal.error or signal.reasoning or "No reasoning supplied."
            lines.append(
                f"  - {signal.agent_id:<18} {direction:<8} {confidence:<9} {shorten(reasoning)}"
            )

    ok = sum(1 for signal in recommendation.signals if signal.status == "ok")
    abstained = sum(1 for signal in recommendation.signals if signal.status == "abstained")
    failed = sum(1 for signal in recommendation.signals if signal.status == "failed")
    lines.extend([
        "",
        f"{len(recommendation.signals)} signals: {ok} ok, {abstained} abstained, {failed} failed",
        f"Saved: {runs_root() / recommendation.run_id}/",
    ])
    return "\n".join(lines)


def ticker_details_text(recommendation: Recommendation, ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in recommendation.tickers:
        return f"{ticker} was not part of run {recommendation.run_id}."

    consensus = recommendation.consensus.get(ticker)
    decision = recommendation.decisions.per_ticker.get(ticker)
    lines = [f"{ticker} details from run {recommendation.run_id}", ""]
    if consensus:
        lines.append(
            "Consensus: "
            f"{field_value(consensus, 'direction', 'n/a')} "
            f"({field_value(consensus, 'confidence', 0.0):.0%})"
        )
    if decision:
        lines.append(
            "Decision: "
            f"{field_value(decision, 'action', 'hold').upper()} "
            f"({field_value(decision, 'confidence', 0.0):.0%})"
        )
        reasoning = field_value(decision, "reasoning", "")
        if reasoning:
            lines.append(f"Reason: {reasoning}")

    lines.extend(["", "Analyst reasoning"])
    for signal in [s for s in recommendation.signals if s.ticker == ticker]:
        status = signal.status
        direction = signal.direction.upper()
        confidence = f"{signal.confidence:.0%}" if status == "ok" else status
        reasoning = signal.error or signal.reasoning or "No reasoning supplied."
        lines.append(f"- {signal.agent_id}: {direction} {confidence}")
        lines.append(f"  {shorten(reasoning, 420)}")
    return "\n".join(lines)


def agent_label(agent_id: str) -> str:
    return agent_id.replace("_", " ").title()


def top_signal_names(signals: list, direction: str, limit: int = 2) -> str:
    picked = sorted(
        [signal for signal in signals if signal.status == "ok" and signal.direction == direction],
        key=lambda signal: signal.confidence,
        reverse=True,
    )[:limit]
    return ", ".join(f"{agent_label(signal.agent_id)} ({signal.confidence:.0%})" for signal in picked)


def notable_data_gap(signals: list) -> str:
    for signal in signals:
        reason = signal.error or signal.reasoning
        low = reason.lower() if reason else ""
        if signal.status in {"failed", "abstained"} or any(term in low for term in ("missing", "unavailable", "no data")):
            return f"{agent_label(signal.agent_id)}: {shorten(reason or signal.status, 140)}"
    return ""


def agent_thoughts_text(signals: list, limit: int = 5) -> str:
    ok = [signal for signal in signals if signal.status == "ok"]
    if not ok:
        return ""
    ordered = sorted(ok, key=lambda signal: signal.confidence, reverse=True)[:limit]
    thoughts = []
    for signal in ordered:
        reason = str(signal.reasoning or field_value(signal, "explain_reasoning", "") or "")
        if not reason:
            reason = f"{signal.direction} at {signal.confidence:.0%} confidence"
        thoughts.append(
            f"{agent_label(signal.agent_id)} was {signal.direction} at {signal.confidence:.0%}: {reason}"
        )
    return "; ".join(thoughts)


def recommendation_verdict_text(recommendation: Recommendation) -> str:
    """Plain-English run verdict derived from the structured recommendation."""
    sections: list[str] = []
    for ticker in recommendation.tickers:
        decision = recommendation.decisions.per_ticker.get(ticker)
        consensus = recommendation.consensus.get(ticker)
        ticker_signals = [signal for signal in recommendation.signals if signal.ticker == ticker]
        if not decision:
            continue

        action = str(field_value(decision, "action", "hold")).upper()
        decision_conf = float(field_value(decision, "confidence", 0.0))
        consensus_dir = str(field_value(consensus, "direction", "n/a")).upper()
        consensus_conf = float(field_value(consensus, "confidence", decision_conf))
        reasoning = shorten(str(field_value(decision, "reasoning", "") or ""), 180)

        paragraphs = [
            (
                f"**{ticker}: {action}.** The analyst consensus is **{consensus_dir}** "
                f"at {consensus_conf:.0%}, and the final decision confidence is {decision_conf:.0%}."
            )
        ]

        if reasoning:
            paragraphs.append(f"Decision note: {reasoning}")

        driver_parts: list[str] = []
        bullish = top_signal_names(ticker_signals, "bullish")
        bearish = top_signal_names(ticker_signals, "bearish")
        if bullish:
            driver_parts.append(f"bullish drivers were {bullish}")
        if bearish:
            driver_parts.append(f"bearish drivers were {bearish}")
        if driver_parts:
            paragraphs.append("Signal read: " + "; ".join(driver_parts) + ".")

        thoughts = agent_thoughts_text(ticker_signals)
        if thoughts:
            paragraphs.append("What the agents thought: " + thoughts + ".")

        gap = notable_data_gap(ticker_signals)
        if gap:
            paragraphs.append(f"Data gap: {gap}.")

        sections.append("\n\n".join(paragraphs))

    plan = recommendation.decisions.hedge_plan
    if plan.pairs:
        pairs = ", ".join(f"{pair.long_ticker}/{pair.short_ticker}" for pair in plan.pairs)
        sections.append(f"**Hedge plan:** paired exposure across {pairs}.")
    elif plan.blocked_longs:
        sections.append(
            f"**Hedge plan:** blocked {', '.join(plan.blocked_longs)} because no short hedge candidate was available."
        )

    if sections:
        sections.append(
            "Use `details <ticker>` for the analyst-by-analyst reasoning. For short-vs-put choice, this run gives the signal layer; timeframe, borrow cost, IV/skew, and strike/expiry still decide the instrument."
        )
    return "\n\n".join(sections) if sections else "Run finished, but no ticker decisions were produced."


def section_header(label: str) -> Text:
    head = Text()
    head.append("▾ ", style=f"bold {TEAL}")
    head.append(label.upper(), style=f"bold {TEAL}")
    return head


def signal_summary_text(signals: list) -> str:
    ok = [signal for signal in signals if signal.status == "ok"]
    if not ok:
        abstained = sum(1 for signal in signals if signal.status == "abstained")
        failed = sum(1 for signal in signals if signal.status == "failed")
        return f"No active votes ({abstained} abstained, {failed} failed)."
    bull = sum(1 for signal in ok if signal.direction == "bullish")
    bear = sum(1 for signal in ok if signal.direction == "bearish")
    neutral = sum(1 for signal in ok if signal.direction == "neutral")
    strongest = sorted(ok, key=lambda signal: signal.confidence, reverse=True)[:2]
    drivers = ", ".join(
        f"{agent_label(signal.agent_id)} {signal.direction} {signal.confidence:.0%}"
        for signal in strongest
    )
    return f"{bull} bull / {bear} bear / {neutral} neutral" + (f" · {drivers}" if drivers else "")


def recommendation_renderable(recommendation: Recommendation) -> Panel:
    """The run as a compact result card. Full analyst reasoning lives in details."""
    sections: list = []

    sections.append(section_header("decisions"))
    table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, padding=(0, 1), expand=True)
    table.add_column("Ticker", style=f"bold {TEAL}", no_wrap=True, width=8)
    table.add_column("Consensus", no_wrap=True, width=11)
    table.add_column("Action", no_wrap=True, width=8)
    table.add_column("Conf", justify="right", no_wrap=True, width=6)
    table.add_column("Why", style=INK, ratio=1, overflow="fold")
    for ticker in recommendation.tickers:
        decision = recommendation.decisions.per_ticker.get(ticker)
        if not decision:
            continue
        consensus_text = field_value(recommendation.consensus.get(ticker), "direction", "n/a")
        action = field_value(decision, "action", "hold")
        confidence = float(field_value(decision, "confidence", 0.0))
        reasoning = field_value(decision, "reasoning", "")
        table.add_row(
            escape(ticker),
            f"[{status_color(consensus_text)}]{consensus_text.upper()}[/{status_color(consensus_text)}]",
            f"[bold {status_color(action)}]{action.upper()}[/bold {status_color(action)}]",
            f"{confidence:.0%}",
            escape(reasoning or ""),
        )
    sections.append(table)

    plan = recommendation.decisions.hedge_plan
    if plan.pairs or plan.blocked_longs:
        sections.append(Text())
        sections.append(section_header("hedge plan"))
        for pair in plan.pairs:
            row = Text("  ▸ ", style=TEAL)
            row.append("LONG ", style=f"bold {SAGE}")
            row.append(f"{pair.long_quantity} {pair.long_ticker}", style=INK)
            row.append("  vs  ", style=ASH)
            row.append("SHORT ", style=f"bold {ROSE}")
            row.append(f"{pair.short_quantity} {pair.short_ticker}", style=INK)
            sections.append(row)
        if plan.blocked_longs:
            sections.append(Text(f"  ▸ blocked (no short hedge): {', '.join(plan.blocked_longs)}", style=AMBER))

    sections.append(Text())
    sections.append(section_header("signal snapshot"))
    signal_table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False, padding=(0, 1), expand=True)
    signal_table.add_column("Ticker", style=f"bold {TEAL}", no_wrap=True, width=8)
    signal_table.add_column("Analyst mix", style=INK, ratio=1, overflow="fold")
    for ticker in recommendation.tickers:
        ticker_signals = [s for s in recommendation.signals if s.ticker == ticker]
        if ticker_signals:
            signal_table.add_row(escape(ticker), escape(signal_summary_text(ticker_signals)))
    sections.append(signal_table)

    ok = sum(1 for signal in recommendation.signals if signal.status == "ok")
    abstained = sum(1 for signal in recommendation.signals if signal.status == "abstained")
    failed = sum(1 for signal in recommendation.signals if signal.status == "failed")
    sections.append(Text())
    sections.append(
        Text.from_markup(
            f"[{ASH}]{len(recommendation.signals)} signals: "
            f"[{SAGE}]{ok} ok[/{SAGE}], "
            f"[{AMBER}]{abstained} abstained[/{AMBER}], "
            f"[{ROSE}]{failed} failed[/{ROSE}]  "
            f"· use [{TEAL}]details <ticker>[/{TEAL}] for full reasoning[/{ASH}]"
        )
    )

    return Panel(
        Group(*sections),
        title=f"[bold {TEAL}]run {escape(recommendation.run_id)}[/bold {TEAL}]",
        title_align="left",
        border_style=ASH,
        padding=(1, 2),
    )


def answer_renderable(text: str):
    try:
        return Markdown(text, style=INK)
    except Exception:
        return Text(text, style=INK)


def render_recommendation(console: Console, recommendation: Recommendation) -> None:
    console.print()
    console.print(recommendation_renderable(recommendation))
    console.print()


def event_renderable(kind: str, title: str, body: str = "") -> Group:
    head = Text()
    if kind == "user":
        head.append("› ", style=f"bold {TEAL}")
        head.append(title, style=f"bold {INK}")
    elif kind == "error":
        head.append("⚠ ", style=f"bold {ROSE}")
        head.append(title, style=ROSE)
    elif kind == "system":
        head.append(title, style=AMBER)
    else:
        head.append(title, style=INK)
    parts: list = [head]
    if body:
        parts.append(Text(body, style=ASH))
    return Group(*parts)


def ticker_details_renderable(recommendation: Recommendation, ticker: str) -> Group:
    ticker = ticker.upper()
    if ticker not in recommendation.tickers:
        return Group(Text(f"{ticker} was not part of run {recommendation.run_id}.", style=AMBER))

    consensus = recommendation.consensus.get(ticker)
    decision = recommendation.decisions.per_ticker.get(ticker)
    head = Text()
    head.append(f"{ticker}", style=f"bold {TEAL}")
    head.append(" details", style=ASH)
    parts: list = [head, Text()]

    summary = Table(box=box.SQUARE, show_edge=True, pad_edge=True, padding=(0, 1), expand=True)
    summary.add_column("Field", style=f"bold {TEAL}", width=14, no_wrap=True)
    summary.add_column("Value", style=INK, ratio=1, overflow="fold")

    if consensus:
        direction = str(field_value(consensus, "direction", "n/a"))
        conf = float(field_value(consensus, "confidence", 0.0))
        summary.add_row("Consensus", f"[bold {status_color(direction)}]{direction.upper()}[/bold {status_color(direction)}]  {conf:.0%}")
    if decision:
        action = str(field_value(decision, "action", "hold"))
        conf = float(field_value(decision, "confidence", 0.0))
        summary.add_row("Decision", f"[bold {status_color(action)}]{action.upper()}[/bold {status_color(action)}]  {conf:.0%}")
        reasoning = field_value(decision, "reasoning", "")
        if reasoning:
            summary.add_row("Reason", escape(reasoning))
    parts.append(summary)

    parts.append(Text())
    parts.append(Text("analyst reasoning", style=f"bold {TEAL}"))
    table = Table(box=box.SQUARE, show_edge=True, pad_edge=True, padding=(0, 1), expand=True)
    table.add_column("Agent", style=f"bold {TEAL}", width=20, no_wrap=True)
    table.add_column("Signal", width=10, no_wrap=True)
    table.add_column("Conf", width=10, no_wrap=True)
    table.add_column("Reasoning", style=INK, ratio=1, overflow="fold")
    for signal in [s for s in recommendation.signals if s.ticker == ticker]:
        status = signal.status
        direction = signal.direction.upper()
        confidence = f"{signal.confidence:.0%}" if status == "ok" else status
        reasoning = signal.error or signal.reasoning or "No reasoning supplied."
        color = status_color(signal.direction)
        table.add_row(
            escape(agent_label(signal.agent_id)),
            f"[bold {color}]{direction}[/bold {color}]",
            escape(confidence),
            escape(shorten(reasoning, 420)),
        )
    parts.append(table)
    return Group(*parts)
