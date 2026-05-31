"""Insider sentiment analyst on the new contract.

Input:  InsiderView (ticker + insider trades).
Output: Signal.

Counts insider buys vs sells (by transaction_shares sign) to produce a
bullish/bearish/neutral signal. This replaces the legacy 'sentiment'
agent's insider-trading portion. The news portion of the old combined
sentiment agent now lives in the dedicated news_sentiment agent.
"""

from __future__ import annotations

import logging

from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import InsiderView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "insider_sentiment"


def run_insider_sentiment_agent(ctx: RunContext, view: InsiderView, recorder=None) -> Signal:
    if not view.trades:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker, reason="No insider trades in window",
        )

    progress.update_status(AGENT_ID, view.ticker, f"Analyzing {len(view.trades)} insider trades")

    try:
        bullish = 0
        bearish = 0
        for t in view.trades:
            shares = getattr(t, "transaction_shares", None)
            if shares is None:
                continue
            if shares > 0:
                bullish += 1
            elif shares < 0:
                bearish += 1
    except Exception as e:
        logger.exception("insider_sentiment failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    total = bullish + bearish
    if total == 0:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker, reason="No classifiable insider trades",
        )

    net = (bullish - bearish) / total
    if net > 0.10:
        direction = "bullish"
    elif net < -0.10:
        direction = "bearish"
    else:
        direction = "neutral"
    confidence = min(0.95, abs(net) + 0.5)

    progress.update_status(AGENT_ID, view.ticker, "Done")
    return Signal(
        agent_id=AGENT_ID,
        ticker=view.ticker,
        direction=direction,
        confidence=confidence,
        reasoning=(
            f"{bullish} insider buys vs {bearish} sells "
            f"(net signal {net:+.0%})"
        ),
        key_factors=[
            f"buys={bullish}",
            f"sells={bearish}",
            f"net={net:+.2f}",
        ],
    )
