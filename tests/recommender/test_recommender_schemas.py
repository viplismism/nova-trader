import pytest

from src.recommender import (
    AgentOpinion,
    Conviction,
    EvidenceRef,
    HedgeRecommendation,
    RecommendationAction,
    RecommendationResult,
    RecommendationRisk,
)
from src.router import TimeHorizon


def test_recommendation_normalizes_ticker_and_exports_legacy_decision():
    result = RecommendationResult(
        ticker="nvda",
        action=RecommendationAction.BUY,
        conviction=Conviction.MEDIUM,
        confidence=0.68,
        horizon=TimeHorizon.MEDIUM_TERM,
        suggested_position_size_pct=3.5,
        summary="Positive growth setup with valuation risk.",
        key_factors=["Revenue growth remains strong"],
        risks=[
            RecommendationRisk(
                name="Valuation compression",
                severity=Conviction.MEDIUM,
                detail="Multiple is sensitive to growth expectations.",
            )
        ],
        evidence=[
            EvidenceRef(
                id="metric:revenue_growth",
                source="financial_metrics",
                summary="Revenue growth is positive.",
                weight=0.8,
            )
        ],
        hedge=HedgeRecommendation(
            short_ticker="amd",
            hedge_ratio=1.0,
            rationale="Pair the long with a weaker semiconductor short.",
        ),
    )

    assert result.ticker == "NVDA"
    assert result.hedge.short_ticker == "AMD"
    assert result.to_trade_decision() == {
        "action": "buy",
        "confidence": 68,
        "reasoning": "Positive growth setup with valuation risk.",
    }


def test_agent_opinion_normalizes_ticker_and_signal():
    opinion = AgentOpinion(
        agent_id="valuation_agent",
        ticker="msft",
        signal=" Bullish ",
        confidence=0.74,
        evidence_ids=["valuation:dcf"],
    )

    assert opinion.ticker == "MSFT"
    assert opinion.signal == "bullish"


def test_buy_recommendation_requires_short_hedge():
    with pytest.raises(ValueError, match="short hedge"):
        RecommendationResult(
            ticker="nvda",
            action=RecommendationAction.BUY,
            conviction=Conviction.MEDIUM,
            confidence=0.68,
            horizon=TimeHorizon.MEDIUM_TERM,
            summary="Positive growth setup.",
        )
