"""Technical analyst on the new contract.

Input:  PriceView (ticker + price history).
Output: Signal (bullish/bearish/neutral + confidence + reasoning).

Math is reused from src/agents/technicals.py to avoid drift.
"""

from __future__ import annotations

import logging

from src.agents.math.technicals import (
    calculate_mean_reversion_signals,
    calculate_momentum_signals,
    calculate_stat_arb_signals,
    calculate_trend_signals,
    calculate_volatility_signals,
    weighted_signal_combination,
)
from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import PriceView
from src.tools.api import prices_to_df
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "technical"

STRATEGY_WEIGHTS = {
    "trend": 0.25,
    "mean_reversion": 0.20,
    "momentum": 0.25,
    "volatility": 0.15,
    "stat_arb": 0.15,
}


def run_technical_agent(ctx: RunContext, view: PriceView, recorder=None) -> Signal:
    """Run the 5-strategy ensemble on the price view."""
    if not view.prices:
        return Signal.abstained(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            reason="No price data available",
        )

    progress.update_status(AGENT_ID, view.ticker, "Computing trend/momentum/vol")

    try:
        prices_df = prices_to_df(view.prices)
        signals = {
            "trend": calculate_trend_signals(prices_df),
            "mean_reversion": calculate_mean_reversion_signals(prices_df),
            "momentum": calculate_momentum_signals(prices_df),
            "volatility": calculate_volatility_signals(prices_df),
            "stat_arb": calculate_stat_arb_signals(prices_df),
        }
        combined = weighted_signal_combination(signals, STRATEGY_WEIGHTS)
    except Exception as e:
        logger.exception("technical agent failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    direction = combined["signal"]
    confidence = float(combined["confidence"])  # already 0..1

    key_factors = []
    for name, sub in signals.items():
        sub_signal = sub.get("signal", "neutral")
        sub_conf = sub.get("confidence", 0)
        key_factors.append(f"{name}={sub_signal} ({sub_conf:.0%})")

    progress.update_status(AGENT_ID, view.ticker, "Done")
    return Signal(
        agent_id=AGENT_ID,
        ticker=view.ticker,
        direction=direction,
        confidence=confidence,
        reasoning=f"Ensemble of 5 strategies: {direction} @ {confidence:.0%}",
        key_factors=key_factors,
    )
