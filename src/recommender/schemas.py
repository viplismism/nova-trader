"""Typed recommendation contracts.

These models separate computed recommendation data from final natural-language
presentation. The LLM can explain these objects, but the product should be able
to evaluate, backtest, and audit them without parsing prose.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Literal

from src.router.schemas import TimeHorizon


class RecommendationAction(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    SHORT = "short"
    COVER = "cover"


class Conviction(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EvidenceRef(BaseModel):
    """Pointer to a structured fact used in a recommendation."""

    id: str
    source: str
    summary: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class RecommendationRisk(BaseModel):
    name: str
    severity: Conviction = Conviction.MEDIUM
    detail: str


class HedgeRecommendation(BaseModel):
    """Paired hedge required for equity long/short recommendations."""

    required: bool = True
    status: Literal["proposed", "missing", "not_required"] = "proposed"
    short_ticker: str | None = None
    hedge_ratio: float | None = Field(default=1.0, ge=0.0)
    rationale: str = ""

    @field_validator("short_ticker")
    @classmethod
    def normalize_short_ticker(cls, ticker: str | None) -> str | None:
        return ticker.strip().upper() if ticker else None


class AgentOpinion(BaseModel):
    agent_id: str
    ticker: str
    signal: str
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: TimeHorizon = TimeHorizon.UNSPECIFIED
    key_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, ticker: str) -> str:
        return ticker.strip().upper()

    @field_validator("signal")
    @classmethod
    def normalize_signal(cls, signal: str) -> str:
        return signal.strip().lower()


class RecommendationResult(BaseModel):
    ticker: str
    action: RecommendationAction
    conviction: Conviction
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: TimeHorizon = TimeHorizon.UNSPECIFIED
    suggested_position_size_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    summary: str
    key_factors: list[str] = Field(default_factory=list)
    risks: list[RecommendationRisk] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    agent_opinions: list[AgentOpinion] = Field(default_factory=list)
    hedge: HedgeRecommendation | None = None
    what_would_change_our_mind: list[str] = Field(default_factory=list)
    model_notes: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, ticker: str) -> str:
        return ticker.strip().upper()

    @model_validator(mode="after")
    def require_short_hedge_for_opening_longs(self):
        if self.action in {RecommendationAction.STRONG_BUY, RecommendationAction.BUY}:
            if not self.hedge or self.hedge.status != "proposed" or not self.hedge.short_ticker:
                raise ValueError("Buy recommendations require a proposed short hedge in equity long/short mode")
        return self

    def to_trade_decision(self) -> dict:
        """Convert recommendation to the legacy portfolio decision shape."""
        action_map = {
            RecommendationAction.STRONG_BUY: "buy",
            RecommendationAction.BUY: "buy",
            RecommendationAction.HOLD: "hold",
            RecommendationAction.SELL: "sell",
            RecommendationAction.STRONG_SELL: "sell",
            RecommendationAction.SHORT: "short",
            RecommendationAction.COVER: "cover",
        }
        return {
            "action": action_map[self.action],
            "confidence": round(self.confidence * 100),
            "reasoning": self.summary,
        }
