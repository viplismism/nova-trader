"""News sentiment agent on the new contract.

Input:  NewsSentimentView (ticker + news + prices).
Output: Signal.

Aggregates per-headline sentiment (vendor sentiment if present, otherwise
keyword fallback) into a direction + confidence. Prices are available for
future "did the headline already move the price" checks; not used in Round 1.
"""

from __future__ import annotations

import logging

from src.agents.math.news_sentiment import _headline_sentiment  # reuse keyword fallback
from src.schemas.context import RunContext
from src.schemas.signals import Signal
from src.schemas.views import NewsSentimentView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "news_sentiment"


def _classify(news_item) -> tuple[str, int]:
    """Return (sentiment, confidence 0-100) for one news item."""
    vendor = getattr(news_item, "sentiment", None)
    if vendor and isinstance(vendor, str):
        v = vendor.strip().lower()
        if v in {"positive", "negative", "neutral"}:
            return v, 70
    return _headline_sentiment(getattr(news_item, "title", "") or "")


def run_news_sentiment_agent(ctx: RunContext, view: NewsSentimentView, recorder=None) -> Signal:
    if not view.news:
        return Signal.abstained(
            agent_id=AGENT_ID,
            ticker=view.ticker,
            reason="No news available for window",
        )

    progress.update_status(AGENT_ID, view.ticker, f"Classifying {len(view.news)} headlines")

    try:
        pos_score = 0.0
        neg_score = 0.0
        pos_count = neg_count = neu_count = 0
        for item in view.news:
            sentiment, conf = _classify(item)
            weight = conf / 100.0
            if sentiment == "positive":
                pos_score += weight
                pos_count += 1
            elif sentiment == "negative":
                neg_score += weight
                neg_count += 1
            else:
                neu_count += 1
    except Exception as e:
        logger.exception("news_sentiment failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    total = pos_count + neg_count + neu_count
    if total == 0:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker, reason="No classifiable headlines"
        )

    net = (pos_score - neg_score) / total
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
        reasoning=f"{pos_count} positive / {neg_count} negative / {neu_count} neutral headlines",
        key_factors=[
            f"net_sentiment={net:+.2f}",
            f"sample_size={total}",
        ],
    )
