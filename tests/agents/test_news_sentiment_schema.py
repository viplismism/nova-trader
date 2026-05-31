from src.agents.math.news_sentiment import Sentiment, _headline_sentiment


def test_sentiment_accepts_confidence_score_alias():
    result = Sentiment.model_validate({
        "sentiment": "positive",
        "confidence_score": 85,
    })

    assert result.sentiment == "positive"
    assert result.confidence == 85


def test_headline_sentiment_fallback_is_deterministic():
    sentiment, confidence = _headline_sentiment("Company shares surge after strong profit beat")

    assert sentiment == "positive"
    assert confidence > 50
