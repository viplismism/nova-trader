"""Aggregate per-ticker analyst signals into a Consensus.

The key behavior change vs the old AgentPad.compute_consensus():
failed and abstained signals are EXCLUDED from the math, not silently
treated as 'neutral with 0% confidence'.
"""

from __future__ import annotations

from src.schemas.signals import Consensus, Signal

# S&P STARS analog: map the normalized weighted score [-1, 1] onto a 1–5 rating.
# Thresholds mirror the bullish/bearish direction cut (±0.10) so the star and the
# direction never disagree, with ±0.40 marking strong conviction.
_STARS_LABELS = {5: "Strong Buy", 4: "Buy", 3: "Hold", 2: "Sell", 1: "Strong Sell"}


def _stars(weighted_score: float) -> tuple[int, str]:
    if weighted_score >= 0.40:
        n = 5
    elif weighted_score >= 0.10:
        n = 4
    elif weighted_score > -0.10:
        n = 3
    elif weighted_score > -0.40:
        n = 2
    else:
        n = 1
    return n, _STARS_LABELS[n]


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
                stars=3, stars_label="Hold",
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

        stars, stars_label = _stars(normalized_score)
        consensus[ticker] = Consensus(
            ticker=ticker,
            direction=direction,
            confidence=avg_confidence,
            weighted_score=normalized_score,
            stars=stars,
            stars_label=stars_label,
            bull_count=bull,
            bear_count=bear,
            neutral_count=neutral,
            contributing=contributing,
            abstained=abstained,
            failed=failed,
        )

    return consensus
