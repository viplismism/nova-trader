"""Growth analyst on the new contract.

Input:  FinancialsView (financial metrics + line items + market cap).
Output: Signal.

Reuses growth-trend / valuation / margin / health helpers from the old
src/agents/growth_agent.py to keep the math identical.
"""

from __future__ import annotations

import logging

from src.agents.math.growth_agent import (
    analyze_growth_trends,
    analyze_margin_trends,
    analyze_valuation,
    check_financial_health,
)
from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import FinancialsView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "growth"

WEIGHTS = {
    "growth": 0.45,
    "valuation": 0.25,
    "margins": 0.20,
    "health": 0.10,
}


def run_growth_agent(ctx: RunContext, view: FinancialsView, recorder=None) -> Signal:
    if len(view.metrics) < 4:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker,
            reason=f"Need 4+ periods, got {len(view.metrics)}",
        )

    progress.update_status(AGENT_ID, view.ticker, "Analyzing growth trends")

    try:
        most_recent = view.metrics[0]
        growth_trends = analyze_growth_trends(view.metrics)
        valuation_metrics = analyze_valuation(most_recent)
        margin_trends = analyze_margin_trends(view.metrics)
        financial_health = check_financial_health(most_recent)

        scores = {
            "growth": growth_trends["score"],
            "valuation": valuation_metrics["score"],
            "margins": margin_trends["score"],
            "health": financial_health["score"],
        }
        weighted_score = sum(scores[k] * WEIGHTS[k] for k in scores)
    except Exception as e:
        logger.exception("growth failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    if weighted_score > 0.6:
        direction = "bullish"
    elif weighted_score < 0.4:
        direction = "bearish"
    else:
        direction = "neutral"
    confidence = min(0.95, abs(weighted_score - 0.5) * 2)

    progress.update_status(AGENT_ID, view.ticker, "Done")
    return Signal(
        agent_id=AGENT_ID,
        ticker=view.ticker,
        direction=direction,
        confidence=confidence,
        reasoning=(
            f"Weighted growth score {weighted_score:.2f}: "
            f"growth {scores['growth']:.2f}, valuation {scores['valuation']:.2f}, "
            f"margins {scores['margins']:.2f}, health {scores['health']:.2f}"
        ),
        key_factors=[
            f"growth_trend={scores['growth']:.2f}",
            f"valuation={scores['valuation']:.2f}",
            f"margins={scores['margins']:.2f}",
            f"health={scores['health']:.2f}",
        ],
    )
