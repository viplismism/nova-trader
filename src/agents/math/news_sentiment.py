"""Pure headline-sentiment helpers used by :mod:`src.agents.news_sentiment`."""

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from typing_extensions import Literal


class Sentiment(BaseModel):
    """Represents the sentiment of a news article."""

    model_config = ConfigDict(populate_by_name=True)

    sentiment: Literal["positive", "negative", "neutral"]
    confidence: int = Field(
        description="Confidence 0-100",
        validation_alias=AliasChoices("confidence", "confidence_score"),
    )


_POSITIVE_TERMS = {
    "beat", "beats", "surge", "surges", "jump", "jumps", "rally", "rallies",
    "raise", "raises", "upgrade", "upgraded", "growth", "record", "profit",
    "profits", "strong", "higher", "tops", "outperform",
}

_NEGATIVE_TERMS = {
    "miss", "misses", "fall", "falls", "drop", "drops", "slump", "slumps",
    "cut", "cuts", "downgrade", "downgraded", "weak", "lower", "loss",
    "losses", "lawsuit", "probe", "investigation", "recall", "layoff",
    "layoffs", "underperform",
}


def _headline_sentiment(title: str) -> tuple[str, int]:
    """Small deterministic fallback for headlines without vendor sentiment."""
    words = {w.strip(".,!?;:'\"()[]{}").lower() for w in title.split()}
    positive_hits = len(words & _POSITIVE_TERMS)
    negative_hits = len(words & _NEGATIVE_TERMS)
    if positive_hits > negative_hits:
        return "positive", min(80, 55 + 10 * positive_hits)
    if negative_hits > positive_hits:
        return "negative", min(80, 55 + 10 * negative_hits)
    return "neutral", 50


def _calculate_confidence_score(
    sentiment_confidences: dict,
    company_news: list,
    overall_signal: str,
    bullish_signals: int,
    bearish_signals: int,
    total_signals: int
) -> float:
    """
    Calculate confidence score for a sentiment signal.
    
    Uses a weighted approach combining LLM confidence scores (70%) with 
    signal proportion (30%) when LLM classifications are available.
    
    Args:
        sentiment_confidences: Dictionary mapping news article IDs to confidence scores.
        company_news: List of CompanyNews objects.
        overall_signal: The overall sentiment signal ("bullish", "bearish", or "neutral").
        bullish_signals: Count of bullish signals.
        bearish_signals: Count of bearish signals.
        total_signals: Total number of signals.
        
    Returns:
        Confidence score as a float between 0 and 100.
    """
    if total_signals == 0:
        return 0.0
    
    # Calculate weighted confidence using LLM confidence scores when available
    if sentiment_confidences:
        # Get articles that match the overall signal
        matching_articles = [
            news for news in company_news 
            if news.sentiment and (
                (overall_signal == "bullish" and news.sentiment == "positive") or
                (overall_signal == "bearish" and news.sentiment == "negative") or
                (overall_signal == "neutral" and news.sentiment == "neutral")
            )
        ]
        
        # Calculate average confidence from LLM-classified articles that match the signal
        llm_confidences = [
            sentiment_confidences[id(news)] 
            for news in matching_articles 
            if id(news) in sentiment_confidences
        ]
        
        if llm_confidences:
            # Weight: 70% from LLM confidence scores, 30% from signal proportion
            avg_llm_confidence = sum(llm_confidences) / len(llm_confidences)
            signal_proportion = (max(bullish_signals, bearish_signals) / total_signals) * 100
            return round(0.7 * avg_llm_confidence + 0.3 * signal_proportion, 2)
    
    # Fallback to proportion-based confidence
    return round((max(bullish_signals, bearish_signals) / total_signals) * 100, 2)
