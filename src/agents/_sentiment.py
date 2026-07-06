"""Shared finance-tuned sentiment scorer.

Plain VADER knows nothing about retail-trading slang — "to the moon, calls
printing" scores 0.0 out of the box. We overlay a finance lexicon (WSB slang
plus options/flow terms) on a single shared analyzer so every social-sentiment
consumer scores text the same way. Also houses the engagement weighting and
mood bucketing so the thresholds live in exactly one place.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Merged from the reddit VaderBackend lexicon and the community-sentiment
# extras (breakout). Values are VADER valence units (roughly -4..+4).
FINANCE_LEXICON: dict[str, float] = {
    "moon": 3.0, "rocket": 2.5, "tendies": 2.5, "calls": 1.5, "long": 1.0,
    "bull": 1.5, "bullish": 2.0, "buy": 1.5, "squeeze": 1.5, "hold": 1.0,
    "breakout": 1.5,
    "puts": -1.5, "short": -1.0, "bear": -1.5, "bearish": -2.0, "sell": -1.5,
    "bagholder": -2.5, "rug": -2.5, "dump": -2.0, "drill": -2.0,
}

_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        analyzer = SentimentIntensityAnalyzer()
        analyzer.lexicon.update(FINANCE_LEXICON)
        _analyzer = analyzer
    return _analyzer


def score_text(text: str) -> float:
    """VADER compound score in [-1, 1] with the finance lexicon applied."""
    return _get_analyzer().polarity_scores(text or "")["compound"]


def engagement_weight(score: int, num_comments: int) -> float:
    """Weight a post by engagement: 1.0 + upvotes + comments.

    Upvotes are floored at 0 so a downvoted post never gets negative weight;
    the +1.0 base keeps zero-engagement posts from vanishing entirely.
    """
    return 1.0 + max(score, 0) + num_comments


def weighted_sentiment(items: list[tuple[float, float]]) -> float | None:
    """Weighted mean of (sentiment, weight) pairs; None if nothing to average."""
    total_weight = sum(w for _, w in items)
    if not items or total_weight <= 0:
        return None
    return sum(s * w for s, w in items) / total_weight


def is_scoreable(text: str) -> bool:
    """True when the text is English-dominant enough for VADER to read.

    VADER is English-only — a Chinese post scores exactly 0.0, which silently
    dilutes the weighted mean toward neutral (a false-neutral, not a real one).
    Heuristic: at least 60% of the alphabetic characters must be ASCII letters.
    """
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if c.isascii())
    return ascii_letters / len(letters) >= 0.6


def mood(score: float | None) -> str:
    """Bucket a sentiment score: bullish >= 0.15, bearish <= -0.15, else neutral."""
    if score is None:
        return "neutral"
    if score >= 0.15:
        return "bullish"
    if score <= -0.15:
        return "bearish"
    return "neutral"
