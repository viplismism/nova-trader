"""Tests for the shared finance-tuned sentiment scorer.

Pure lexicon math — no network, no monkeypatching needed.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.agents import _sentiment
from src.agents._sentiment import (
    engagement_weight,
    mood,
    score_text,
    weighted_sentiment,
)


def test_slang_positive_where_plain_vader_is_flat():
    text = "to the moon, calls printing, rocket"
    # The whole point of the overlay: stock VADER sees nothing here.
    assert SentimentIntensityAnalyzer().polarity_scores(text)["compound"] == 0.0
    assert score_text(text) > 0.15


def test_slang_negative():
    assert score_text("bagholders getting rugged, puts") < -0.15


def test_score_text_handles_empty():
    assert score_text("") == 0.0


def test_shared_analyzer_instance():
    score_text("warm up")
    first = _sentiment._analyzer
    score_text("again")
    assert _sentiment._analyzer is first


def test_engagement_weight():
    assert engagement_weight(10, 5) == 16.0
    assert engagement_weight(0, 0) == 1.0
    # Downvoted posts floor at 0 upvotes, keeping base + comments.
    assert engagement_weight(-50, 3) == 4.0


def test_weighted_sentiment_math():
    assert weighted_sentiment([(1.0, 1.0), (0.0, 3.0)]) == 0.25
    assert weighted_sentiment([(0.5, 2.0)]) == 0.5


def test_weighted_sentiment_degenerate_cases():
    assert weighted_sentiment([]) is None
    assert weighted_sentiment([(0.9, 0.0)]) is None


def test_mood_thresholds_inclusive():
    assert mood(0.15) == "bullish"
    assert mood(-0.15) == "bearish"
    assert mood(0.1499) == "neutral"
    assert mood(-0.1499) == "neutral"
    assert mood(0.0) == "neutral"
    assert mood(None) == "neutral"
