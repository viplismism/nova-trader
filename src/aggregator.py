"""Aggregate per-ticker analyst signals into a Consensus.

The key behavior change vs the old AgentPad.compute_consensus():
failed and abstained signals are EXCLUDED from the math, not silently
treated as 'neutral with 0% confidence'.
"""

from __future__ import annotations

from src.schemas.signals import Consensus, Signal


def compute_consensus(signals: list[Signal], tickers: list[str]) -> dict[str, Consensus]:
    """Build one Consensus per ticker from the signals list."""
    consensus: dict[str, Consensus] = {}

    by_ticker: dict[str, list[Signal]] = {t: [] for t in tickers}
    for s in signals:
        by_ticker.setdefault(s.ticker, []).append(s)

    for ticker, ticker_signals in by_ticker.items():
        contributing: list[str] = []
        abstained: list[str] = []
        failed: list[str] = []
        bull = bear = neutral = 0
        weighted_score = 0.0
        total_confidence = 0.0

        for s in ticker_signals:
            if s.status == "failed":
                failed.append(s.agent_id)
                continue
            if s.status == "abstained":
                abstained.append(s.agent_id)
                continue
            contributing.append(s.agent_id)
            total_confidence += s.confidence
            if s.direction == "bullish":
                bull += 1
                weighted_score += s.confidence
            elif s.direction == "bearish":
                bear += 1
                weighted_score -= s.confidence
            else:
                neutral += 1

        total = len(contributing)
        if total == 0:
            consensus[ticker] = Consensus(
                ticker=ticker,
                direction="neutral",
                confidence=0.0,
                weighted_score=0.0,
                bull_count=0, bear_count=0, neutral_count=0,
                contributing=contributing,
                abstained=abstained,
                failed=failed,
            )
            continue

        avg_confidence = total_confidence / total
        normalized_score = weighted_score / total  # bounded [-1, 1]
        if normalized_score > 0.10:
            direction = "bullish"
        elif normalized_score < -0.10:
            direction = "bearish"
        else:
            direction = "neutral"

        consensus[ticker] = Consensus(
            ticker=ticker,
            direction=direction,
            confidence=avg_confidence,
            weighted_score=normalized_score,
            bull_count=bull,
            bear_count=bear,
            neutral_count=neutral,
            contributing=contributing,
            abstained=abstained,
            failed=failed,
        )

    return consensus
