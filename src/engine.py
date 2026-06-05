"""The compact runtime.

Pipeline:
    1. Build snapshot (one set of API calls).
    2. For each (agent, ticker) pair: slice view -> run agent -> collect Signal.
    3. Aggregate signals into Consensus per ticker.
    4. Risk manager: PortfolioView -> Limits.
    5. Portfolio manager: (PortfolioView with consensus, Limits) -> Decisions.
    6. Return Recommendation.

No state dict, no messages, no graph. Just typed handoffs.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.runs import RunRecorder
from src.schemas.context import RunContext
from src.schemas.signals import Recommendation, Signal
from src.schemas.snapshot import MarketSnapshot
from src.utils.progress import progress
from src.agents.portfolio import run_portfolio_manager
from src.agents.risk import run_risk_manager
from src.aggregator import compute_consensus
from src.registry import AGENT_REGISTRY, AgentSpec
from src.slicer import build_portfolio_view, build_view
from src.snapshot import build_snapshot

logger = logging.getLogger(__name__)


def _run_one_agent(
    spec: AgentSpec,
    ctx: RunContext,
    snapshot: MarketSnapshot,
    ticker: str,
    recorder: RunRecorder | None,
) -> Signal:
    """Slice the right view, call the agent. Errors become failed Signals."""
    try:
        view = build_view(spec.view_class, snapshot, ticker)
        if recorder is not None:
            recorder.append_view(spec.agent_id, ticker, view)
    except Exception as e:
        logger.exception("view build failed for %s on %s", spec.agent_id, ticker)
        return Signal.failed(agent_id=spec.agent_id, ticker=ticker, error=f"view build: {e}")

    try:
        signal = spec.runner(ctx, view, recorder)
    except Exception as e:
        logger.exception("agent %s failed for %s", spec.agent_id, ticker)
        return Signal.failed(agent_id=spec.agent_id, ticker=ticker, error=str(e))

    # Explain-only enrichment — narrates the deterministic numbers without touching the
    # verdict. No-op (zero LLM calls) when the run toggle is off. Runs inside this pool
    # task so calls overlap sibling agents.
    from src.agents.explain import add_explain_reasoning

    return add_explain_reasoning(signal, spec, view, ctx, recorder)


def run_engine(
    ctx: RunContext,
    selected_agents: list[str] | None = None,
    api_key: str | None = None,
    max_workers: int = 8,
    *,
    snapshot: MarketSnapshot | None = None,
    record: bool = True,
) -> Recommendation:
    """Full pipeline: snapshot -> agents -> consensus -> risk -> portfolio.

    Args:
        snapshot: Pre-built MarketSnapshot. When provided (e.g. from `rerun`),
            no API calls are made — the run replays against this exact data.
        record: If True (default), every step writes to ~/.nova-trader/runs/<run_id>/.
            Set False for backtest inner loops where per-day audit files would be noise.
    """

    recorder: RunRecorder | None = RunRecorder(ctx.run_id) if record else None

    if recorder is not None:
        recorder.write_metadata({
            "run_id": ctx.run_id,
            "as_of": ctx.as_of.isoformat(),
            "tickers": list(ctx.tickers),
            "start_date": ctx.start_date,
            "end_date": ctx.end_date,
            "model": ctx.request.model.model_dump(),
            "portfolio_mode": ctx.request.portfolio_mode,
            "seed": ctx.derive_seed(),
            "code_sha": ctx.code_sha,
            "selected_agents": selected_agents or list(AGENT_REGISTRY.keys()),
            "replayed_from_snapshot": snapshot is not None,
        })

    # 1. Snapshot once (or use the provided one for replay).
    if snapshot is None:
        snapshot = build_snapshot(ctx, api_key=api_key)
    if recorder is not None:
        recorder.write_snapshot(snapshot)

    # 2. Run analysts in parallel — one task per (agent, ticker) pair.
    if selected_agents is None or not selected_agents:
        agent_ids = list(AGENT_REGISTRY.keys())
    else:
        agent_ids = [a for a in selected_agents if a in AGENT_REGISTRY]
        if not agent_ids:
            raise ValueError(f"No valid agents selected. Got: {selected_agents}")

    signals: list[Signal] = []
    tasks = [
        (AGENT_REGISTRY[aid], ticker)
        for aid in agent_ids
        for ticker in ctx.tickers
    ]
    workers = min(len(tasks), max_workers) or 1

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one_agent, spec, ctx, snapshot, ticker, recorder): (spec.agent_id, ticker)
            for spec, ticker in tasks
        }
        for fut in as_completed(futures):
            sig = fut.result()
            signals.append(sig)
            if recorder is not None:
                recorder.append_signal(sig)
            # Surface each agent's stance for the live agentic view (chatter feed).
            takeaway = sig.direction if sig.status == "ok" else sig.status
            progress.update_status(sig.agent_id, sig.ticker, "done", analysis=takeaway)

    # 3. Aggregate into consensus.
    consensus = compute_consensus(signals, ctx.tickers)

    # 4. Risk manager.
    portfolio_view = build_portfolio_view(snapshot, ctx.request.portfolio, consensus)
    limits = run_risk_manager(ctx, portfolio_view)

    # 5. Portfolio manager.
    decisions = run_portfolio_manager(ctx, portfolio_view, limits)

    # 6. Council reasoning — explain-only narratives over the whole board. No-op
    # (zero LLM calls) when /reasoning is off. Never alters limits/decisions.
    from src.agents.explain import add_council_reasoning

    risk_reasoning, portfolio_reasoning = add_council_reasoning(
        ctx, signals, consensus, limits, decisions, recorder
    )

    recommendation = Recommendation(
        run_id=ctx.run_id,
        as_of=ctx.as_of.isoformat(),
        tickers=list(ctx.tickers),
        signals=signals,
        consensus=consensus,
        limits=limits,
        decisions=decisions,
        summary=_build_summary(consensus, decisions),
        risk_reasoning=risk_reasoning,
        portfolio_reasoning=portfolio_reasoning,
    )

    if recorder is not None:
        recorder.write_recommendation(recommendation)

    return recommendation


def _build_summary(consensus: dict, decisions) -> str:
    parts = []
    for ticker, dec in decisions.per_ticker.items():
        c = consensus.get(ticker)
        if c:
            parts.append(
                f"{ticker}: consensus={c.direction} ({c.confidence:.0%}), "
                f"action={dec.action}"
                + (f" x{dec.quantity}" if dec.quantity else "")
            )
        else:
            parts.append(f"{ticker}: action={dec.action}")
    return " | ".join(parts)
