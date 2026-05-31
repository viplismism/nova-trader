"""Warren Buffett agent on the new contract.

Input:  PersonaView (financials + line_items + market_cap + news + insider).
Output: Signal.

Reuses the shared scoring functions and Jinja2 prompt template. The file
pulls scoring inputs from the typed view, calls the LLM, and returns a Signal.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.agents.scoring import SCORING_FUNCTIONS
from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import PersonaView
from src.utils.llm import call_llm
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "warren_buffett"
DISPLAY_NAME = "Warren Buffett"

# Reuse the existing Jinja2 prompt template.
_PROMPTS_DIR = Path(__file__).resolve().parent / "templates" / "prompts"
_jinja = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Buffett's scoring functions.
SCORING = [
    ("roe", {"threshold": 0.15}),
    ("margins", {}),
    ("debt_to_equity", {"max_safe": 0.5}),
    ("current_ratio", {}),
    ("gross_margin_stability", {}),
    ("capital_intensity", {}),
    ("book_value_growth", {}),
    ("share_dilution", {}),
    ("fcf_conversion", {}),
    ("intrinsic_value", {}),
]
WEIGHTS = {
    "roe": 1.0, "margins": 1.0, "debt_to_equity": 1.0, "current_ratio": 0.5,
    "gross_margin_stability": 1.0, "capital_intensity": 0.8,
    "book_value_growth": 1.0, "share_dilution": 0.8,
    "fcf_conversion": 1.0, "intrinsic_value": 1.5,
}
BULLISH_THRESHOLD = 0.70
BEARISH_THRESHOLD = 0.30


class _LLMSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="0-100")
    reasoning: str


def _run_scoring(view: PersonaView) -> dict[str, dict]:
    """Run all Buffett scoring functions against the view."""
    data_map = {
        "metrics": view.metrics,
        "financial_line_items": view.line_items,
        "insider_trades": view.insider,
        "company_news": view.news,
        "market_cap": view.market_cap,
    }
    import inspect
    results: dict[str, dict] = {}
    for func_name, params in SCORING:
        fn = SCORING_FUNCTIONS.get(func_name)
        if fn is None:
            results[func_name] = {"score": 0, "max_score": 0, "details": f"Unknown scoring function: {func_name}"}
            continue
        sig = inspect.signature(fn)
        kwargs = {}
        for param_name in sig.parameters:
            if param_name in data_map:
                kwargs[param_name] = data_map[param_name]
            elif param_name in params:
                kwargs[param_name] = params[param_name]
        try:
            results[func_name] = fn(**kwargs)
        except Exception as e:
            results[func_name] = {"score": 0, "max_score": 0, "details": f"Error in {func_name}: {e}"}
    return results


def _aggregate(results: dict[str, dict]) -> tuple[float, float, str]:
    total = 0.0
    total_max = 0.0
    for name, r in results.items():
        w = WEIGHTS.get(name, 1.0)
        total += r.get("score", 0) * w
        total_max += r.get("max_score", 0) * w
    if total_max <= 0:
        return total, total_max, "neutral"
    normalized = total / total_max
    if normalized >= BULLISH_THRESHOLD:
        return total, total_max, "bullish"
    if normalized <= BEARISH_THRESHOLD:
        return total, total_max, "bearish"
    return total, total_max, "neutral"


def run_warren_buffett_agent(ctx: RunContext, view: PersonaView, recorder=None) -> Signal:
    if not view.metrics and not view.line_items:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker, reason="No financial data"
        )

    progress.update_status(AGENT_ID, view.ticker, "Running Buffett scoring")
    try:
        results = _run_scoring(view)
        score, max_score, hint = _aggregate(results)
    except Exception as e:
        logger.exception("buffett scoring failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    # Build the LLM prompt using the persona template.
    template_ctx = {
        "name": DISPLAY_NAME,
        "philosophy": (
            "Seek companies with durable competitive advantages, consistent earnings, "
            "high ROE, conservative debt, and honest management. Buy at a meaningful "
            "discount to intrinsic value."
        ),
        "focus_metrics": ["ROE", "Debt/Equity", "FCF consistency", "Margin of safety"],
        "signal_rules": (
            "Bullish: strong moat + consistent earnings + margin of safety > 0. "
            "Bearish: poor business quality or clearly overvalued. "
            "Neutral: good business but insufficient margin of safety."
        ),
        "ticker": view.ticker,
        "signal_hint": hint,
        "score": f"{score:.1f}",
        "max_score": f"{max_score:.1f}",
        "score_pct": f"{(score / max_score * 100):.0f}" if max_score > 0 else "0",
        "scoring_results": results,
        "market_cap": view.market_cap,
    }
    system_prompt = _jinja.get_template("default.j2").render(**template_ctx)

    facts_lines = [
        f"- {name}: {r.get('score', 0)}/{r.get('max_score', 0)} — {r.get('details', '')}"
        for name, r in results.items()
    ]
    prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Ticker: {view.ticker}\n"
                f"Score: {score:.1f}/{max_score:.1f} ({template_ctx['score_pct']}%)\n"
                f"Recent news headlines: {len(view.news)} items\n"
                f"Recent insider trades: {len(view.insider)} items\n"
                "Scoring details:\n"
                + "\n".join(facts_lines)
                + "\n\nReturn JSON: {\"signal\": ..., \"confidence\": 0-100, \"reasoning\": \"...\"}"
            ),
        },
    ]

    progress.update_status(AGENT_ID, view.ticker, "Calling LLM")
    try:
        # Build a thin compatibility shim because call_llm reads model config off state["metadata"].
        state_shim = {
            "metadata": {
                "model_name": ctx.request.model.name,
                "model_provider": ctx.request.model.provider,
            }
        }
        llm_out: _LLMSignal = call_llm(
            prompt=prompt,
            pydantic_model=_LLMSignal,
            agent_name=AGENT_ID,
            state=state_shim,  # type: ignore[arg-type]
            default_factory=lambda: _LLMSignal(signal=hint, confidence=50, reasoning="LLM unavailable; using scoring hint"),
            seed=ctx.derive_seed(),
            recorder=recorder,
            ticker=view.ticker,
        )
    except Exception as e:
        logger.exception("buffett LLM call failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    progress.update_status(AGENT_ID, view.ticker, "Done")
    return Signal(
        agent_id=AGENT_ID,
        ticker=view.ticker,
        direction=llm_out.signal,
        confidence=llm_out.confidence / 100.0,
        reasoning=llm_out.reasoning,
        key_factors=[
            f"score={score:.1f}/{max_score:.1f} ({template_ctx['score_pct']}%)",
            f"scoring_hint={hint}",
        ],
    )
