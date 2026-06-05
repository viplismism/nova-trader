"""Explain-only LLM reasoning over a deterministic agent's numbers.

The 6 quant analysts compute a Signal (direction/confidence/key_factors) with only
a terse note. This layer asks an LLM to narrate WHY those numbers produced that read
— without ever changing the verdict. It is verdict-frozen by construction: the model
is only ever asked for a single free-text `explanation` field, and the result is
attached via model_copy, so direction/confidence/status/key_factors are byte-identical.

Gated off entirely (zero LLM calls) when the run toggle is off, the signal isn't a
clean "ok", or the agent already reasons via an LLM (warren_buffett). Any failure
returns the original signal so a run never breaks.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import FinancialsView, InsiderView, NewsSentimentView, PersonaView, PriceView
from src.utils.llm import call_llm

logger = logging.getLogger(__name__)

_LLM_NATIVE = {"warren_buffett"}  # already produces its own LLM reasoning


class _Explanation(BaseModel):
    explanation: str = Field(default="", description="2-4 sentence finance 'why' explaining ONLY the given numbers")


_SYSTEM = (
    "You are a hedge-fund analyst writing the one-paragraph 'why' behind an "
    "already-decided signal. You ONLY explain the numbers you are given.\n"
    "HARD RULES:\n"
    "- The DIRECTION and CONFIDENCE are FINAL. Do not agree, disagree, restate, "
    "hedge, or flip them. Never propose a different signal.\n"
    "- Cite ONLY numbers present in the FACTS block. Invent nothing — no prices, "
    "ratios, dates, or events that are not listed.\n"
    "- Stay strictly finance-scoped. 2-4 sentences, under 80 words. No preamble, "
    "no headings, no bullet points.\n"
    "- Explain WHY these specific factors produced this read."
)


def _view_slice(view) -> str:
    """A tiny, empty-safe corroboration summary of the agent's raw data. Bounded to
    keep tokens + hallucination surface small; never raises (a throw here would be
    swallowed upstream and silently skip narration)."""
    try:
        if isinstance(view, PriceView):
            n = len(view.prices)
            if n >= 2 and view.prices[0].close:
                chg = (view.prices[-1].close / view.prices[0].close - 1) * 100
                return f"price bars={n}; period close move={chg:+.1f}%"
            return f"price bars={n}"
        if isinstance(view, (FinancialsView, PersonaView)):
            return (
                f"metric periods={len(view.metrics)}; "
                f"line-item periods={len(view.line_items)}; market_cap={view.market_cap}"
            )
        if isinstance(view, NewsSentimentView):
            return f"news items={len(view.news)}; price bars={len(view.prices)}"
        if isinstance(view, InsiderView):
            return f"insider trades={len(view.trades)}"
    except Exception:
        logger.debug("view slice failed", exc_info=True)
    return "no extra view summary"


def _build_prompt(spec, view, signal: Signal) -> list[dict]:
    factors = "\n".join(f"- {f}" for f in signal.key_factors) or "- (none)"
    user = (
        f"Analyst: {spec.display_name} ({spec.agent_id})\n"
        f"Ticker: {signal.ticker}\n"
        f"FINAL signal (do not change): {signal.direction} @ {signal.confidence:.0%} confidence\n"
        f"Agent's own note: {signal.reasoning or '(none)'}\n\n"
        f"FACTS (the numbers behind this signal):\n{factors}\n"
        f"View summary: {_view_slice(view)}\n\n"
        'Write the 2-4 sentence explanation. Return JSON: {"explanation": "..."}'
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def add_explain_reasoning(signal: Signal, spec, view, ctx: RunContext, recorder=None) -> Signal:
    """Attach an explain-only narration to a deterministic signal. Returns the signal
    UNCHANGED (and makes zero LLM calls) when gated off, non-ok, or LLM-native."""
    if not ctx.request.show_reasoning:
        return signal
    if signal.status != "ok":
        return signal
    if spec.agent_id in _LLM_NATIVE:
        return signal
    try:
        state_shim = {
            "metadata": {
                "model_name": ctx.request.model.name,
                "model_provider": ctx.request.model.provider,
            }
        }
        out: _Explanation = call_llm(
            prompt=_build_prompt(spec, view, signal),
            pydantic_model=_Explanation,
            agent_name=spec.agent_id,
            state=state_shim,  # type: ignore[arg-type]
            default_factory=lambda: _Explanation(explanation=""),
            seed=ctx.derive_seed(),
            recorder=recorder,
            ticker=signal.ticker,
        )
    except Exception:
        logger.exception("explain reasoning failed for %s/%s", spec.agent_id, signal.ticker)
        return signal
    text = (out.explanation or "").strip()
    if not text:
        return signal
    return signal.model_copy(update={"explain_reasoning": text})


# ── Council reasoning (risk + portfolio managers) ────────────────────────────
# The risk + portfolio managers are not analysts producing one signal each — they
# are the synthesis council that takes ALL analyst signals + consensus + limits and
# makes the final call. So they get a "bigger" narrative reasoning over the whole
# board. Two LLM calls total per run (not per ticker), explain-only, same toggle.

_COUNCIL_SYSTEM = (
    "You are a member of a hedge fund's investment committee writing the committee's "
    "reasoning behind decisions that are ALREADY FINAL. You explain how the council "
    "weighed its inputs — you never change the verdict.\n"
    "HARD RULES:\n"
    "- The numbers, limits, and decisions given are FINAL. Do not flip, re-decide, or "
    "second-guess them. Never propose different actions or sizes.\n"
    "- Cite ONLY the figures in the BOARD block. Invent nothing.\n"
    "- Write how the council reasoned across the names: weigh the analyst agreement, "
    "the conviction, and the risk limits against each other.\n"
    "- Finance-scoped, concrete, 4-8 sentences, under ~150 words. No headings or bullets."
)


def _council_call(ctx: RunContext, agent_id: str, board: str, recorder=None) -> str:
    """One explain-only council narrative. Returns '' on any failure or when gated off."""
    if not ctx.request.show_reasoning:
        return ""
    try:
        state_shim = {
            "metadata": {
                "model_name": ctx.request.model.name,
                "model_provider": ctx.request.model.provider,
            }
        }
        out: _Explanation = call_llm(
            prompt=[
                {"role": "system", "content": _COUNCIL_SYSTEM},
                {"role": "user", "content": board + '\n\nWrite the committee reasoning. Return JSON: {"explanation": "..."}'},
            ],
            pydantic_model=_Explanation,
            agent_name=agent_id,
            state=state_shim,  # type: ignore[arg-type]
            default_factory=lambda: _Explanation(explanation=""),
            seed=ctx.derive_seed(),
            recorder=recorder,
            ticker=None,
        )
    except Exception:
        logger.exception("council reasoning failed for %s", agent_id)
        return ""
    return (out.explanation or "").strip()


def _risk_board(ctx: RunContext, limits) -> str:
    lines = [f"Mode: {getattr(ctx.request, 'portfolio_mode', 'research')}", "Per-ticker sizing the risk manager set:"]
    for tkr, lim in (limits.per_ticker or {}).items():
        lines.append(
            f"- {tkr}: price ${lim.current_price:,.2f}, annualized vol {lim.annualized_volatility:.0%}, "
            f"correlation multiplier {lim.correlation_multiplier:.2f}, "
            f"max position ${lim.max_position_dollars:,.0f} ({lim.max_shares} sh)"
        )
    return "BOARD:\n" + "\n".join(lines)


def _portfolio_board(ctx: RunContext, signals, consensus, limits, decisions) -> str:
    by_ticker: dict[str, list] = {}
    for s in signals:
        if s.status == "ok":
            by_ticker.setdefault(s.ticker, []).append(f"{s.agent_id}={s.direction}({s.confidence:.0%})")
    lines = [f"Mode: {getattr(ctx.request, 'portfolio_mode', 'research')}"]
    for tkr in ctx.tickers:
        con = consensus.get(tkr)
        lim = (limits.per_ticker or {}).get(tkr)
        dec = (decisions.per_ticker or {}).get(tkr)
        stances = ", ".join(by_ticker.get(tkr, [])) or "(no analyst signals)"
        con_str = (
            f"consensus {con.direction} {con.confidence:.0%} "
            f"({con.bull_count} bull / {con.bear_count} bear / {con.neutral_count} neutral)"
            if con else "no consensus"
        )
        lim_str = f"risk cap {lim.max_shares} sh (${lim.max_position_dollars:,.0f})" if lim else "no limit"
        dec_str = (
            f"DECISION {dec.action.upper()}" + (f" x{dec.quantity}" if dec and dec.quantity else "")
            if dec else "no decision"
        )
        lines.append(f"- {tkr}: analysts [{stances}]; {con_str}; {lim_str}; {dec_str}")
    hedge = getattr(decisions, "hedge_plan", None)
    if hedge and getattr(hedge, "status", "not_required") != "not_required":
        lines.append(f"Hedge plan: {hedge.status}")
    return "BOARD:\n" + "\n".join(lines)


def add_council_reasoning(ctx: RunContext, signals, consensus, limits, decisions, recorder=None) -> tuple[str, str]:
    """Generate the risk + portfolio council narratives. ('', '') when gated off."""
    if not ctx.request.show_reasoning:
        return "", ""
    risk_text = _council_call(ctx, "risk_manager", _risk_board(ctx, limits), recorder)
    portfolio_text = _council_call(ctx, "portfolio_manager", _portfolio_board(ctx, signals, consensus, limits, decisions), recorder)
    return risk_text, portfolio_text
