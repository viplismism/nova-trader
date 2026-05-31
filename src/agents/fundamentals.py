"""Fundamentals analyst on the new contract.

Input:  FinancialsView (financial metrics + line items + market cap).
Output: Signal.

Looks at profitability, growth, financial health, and valuation ratios
of the most recent reporting period to produce a bullish/bearish/neutral
signal.
"""

from __future__ import annotations

import logging

from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import FinancialsView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "fundamentals"


def _direction_from_score(score: int, total: int) -> tuple[str, float]:
    """Map a sub-score (0..total) to a direction + confidence."""
    if score >= total * 0.66:
        return "bullish", score / total
    if score == 0:
        return "bearish", 0.7
    return "neutral", 0.5


def run_fundamentals_agent(ctx: RunContext, view: FinancialsView, recorder=None) -> Signal:
    if not view.metrics:
        return Signal.abstained(agent_id=AGENT_ID, ticker=view.ticker, reason="No financial metrics")

    m = view.metrics[0]
    progress.update_status(AGENT_ID, view.ticker, "Scoring profitability, growth, health, valuation")

    try:
        # 1. Profitability
        profit_thresholds = [
            (m.return_on_equity, 0.15),
            (m.net_margin, 0.20),
            (m.operating_margin, 0.15),
        ]
        profit_score = sum(v is not None and v > t for v, t in profit_thresholds)

        # 2. Growth
        growth_thresholds = [
            (m.revenue_growth, 0.10),
            (m.earnings_growth, 0.10),
            (m.book_value_growth, 0.10),
        ]
        growth_score = sum(v is not None and v > t for v, t in growth_thresholds)

        # 3. Health
        health_score = 0
        if m.current_ratio and m.current_ratio > 1.5:
            health_score += 1
        if m.debt_to_equity is not None and m.debt_to_equity < 0.5:
            health_score += 1
        if (m.free_cash_flow_per_share and m.earnings_per_share and
                m.free_cash_flow_per_share > m.earnings_per_share * 0.8):
            health_score += 1

        # 4. Valuation ratios (lower is more bullish)
        val_score = 0
        if m.price_to_earnings_ratio and m.price_to_earnings_ratio < 25:
            val_score += 1
        if m.price_to_book_ratio and m.price_to_book_ratio < 3:
            val_score += 1
        if m.price_to_sales_ratio and m.price_to_sales_ratio < 5:
            val_score += 1

        # Aggregate
        scores = [profit_score, growth_score, health_score, val_score]
        total_bullish = sum(1 for s in scores if s >= 2)
        total_bearish = sum(1 for s in scores if s == 0)
    except Exception as e:
        logger.exception("fundamentals failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    if total_bullish >= 3:
        direction = "bullish"
        confidence = min(0.95, 0.5 + 0.15 * total_bullish)
    elif total_bearish >= 3:
        direction = "bearish"
        confidence = min(0.95, 0.5 + 0.15 * total_bearish)
    else:
        direction = "neutral"
        confidence = 0.5

    key_factors = [
        f"profitability {profit_score}/3",
        f"growth {growth_score}/3",
        f"health {health_score}/3",
        f"valuation {val_score}/3",
    ]
    reasoning = (
        f"{total_bullish} bullish sub-signals, {total_bearish} bearish "
        f"across profitability/growth/health/valuation"
    )

    progress.update_status(AGENT_ID, view.ticker, "Done")
    return Signal(
        agent_id=AGENT_ID,
        ticker=view.ticker,
        direction=direction,
        confidence=confidence,
        reasoning=reasoning,
        key_factors=key_factors,
    )
