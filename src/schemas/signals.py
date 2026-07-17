"""Agent output schemas.

Every analyst returns a Signal. The aggregator produces a Consensus.
The risk manager produces Limits. The portfolio manager produces Decisions.
The runner produces a Recommendation.

No fake chat messages anywhere.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Ticker = str
SignalStatus = Literal["ok", "abstained", "failed"]
Direction = Literal["bullish", "bearish", "neutral"]
Action = Literal["buy", "sell", "short", "cover", "hold"]


# ── Source citations ────────────────────────────────────────
# Structured provenance so the UI can render clickable evidence chips instead of
# burying URLs / filing refs inside free-text key_factors.


class WebSourceCitation(BaseModel):
    title: str
    url: str
    snippet: str = ""


class FilingCitation(BaseModel):
    chunk_id: str
    form: str
    fiscal_year: str = ""
    item: str = ""
    url: str = ""
    snippet: str = ""


# ── Analyst output ──────────────────────────────────────────


class Signal(BaseModel):
    """The single output of every analyst agent.

    The status field distinguishes a real opinion from a failure.
    The aggregator uses this to EXCLUDE failed/abstained agents from
    consensus — they don't get silently treated as 'neutral' anymore.
    """

    agent_id: str
    ticker: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    # Structured citations surfaced as clickable chips in the UI. Additive and
    # default-empty so older saved runs still validate.
    web_sources: list[WebSourceCitation] = Field(default_factory=list)
    filing_sources: list[FilingCitation] = Field(default_factory=list)
    # LLM explain-only narration over the deterministic numbers. Verdict-immutable:
    # never set by Signal.failed/abstained and never affects direction/confidence.
    explain_reasoning: str = ""
    status: SignalStatus = "ok"
    error: str | None = None

    @classmethod
    def failed(cls, agent_id: str, ticker: str, error: str) -> "Signal":
        """Build a failed signal — direction is neutral but status flags the failure."""
        return cls(
            agent_id=agent_id,
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            reasoning="",
            status="failed",
            error=error,
        )

    @classmethod
    def abstained(cls, agent_id: str, ticker: str, reason: str) -> "Signal":
        """Build an abstain signal — agent saw the data but had no opinion."""
        return cls(
            agent_id=agent_id,
            ticker=ticker,
            direction="neutral",
            confidence=0.0,
            reasoning=reason,
            status="abstained",
        )


# ── Aggregation ─────────────────────────────────────────────


class Consensus(BaseModel):
    """Aggregated view of all analyst signals for ONE ticker.

    Failed and abstained agents are excluded from the math but tracked
    in `failed` / `abstained` for the audit trail.
    """

    ticker: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=-1.0, le=1.0)
    # S&P-STARS-style 1–5 rating derived from weighted_score (5=strong buy … 1=strong sell).
    stars: int = Field(ge=1, le=5, default=3)
    stars_label: str = "Hold"
    bull_count: int = 0
    bear_count: int = 0
    neutral_count: int = 0
    contributing: list[str] = Field(default_factory=list)
    abstained: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


# ── Risk output ─────────────────────────────────────────────


class TickerLimit(BaseModel):
    ticker: str
    current_price: float
    max_position_dollars: float
    max_shares: int
    annualized_volatility: float
    correlation_multiplier: float = 1.0
    remaining_position_limit: float = 0.0


class Limits(BaseModel):
    per_ticker: dict[Ticker, TickerLimit] = Field(default_factory=dict)


# ── Portfolio output ────────────────────────────────────────


class TickerDecision(BaseModel):
    ticker: str
    action: Action
    quantity: int = 0
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str = ""
    hedge_pair_id: str | None = None


class HedgePair(BaseModel):
    pair_id: str
    long_ticker: str
    short_ticker: str
    long_quantity: int
    short_quantity: int
    long_notional: float
    short_notional: float
    hedge_ratio: float


class HedgePlan(BaseModel):
    strategy: Literal["equity_long_short"] = "equity_long_short"
    target_hedge_ratio: float = 1.0
    status: Literal["not_required", "balanced", "partially_hedged", "blocked"] = "not_required"
    pairs: list[HedgePair] = Field(default_factory=list)
    blocked_longs: list[str] = Field(default_factory=list)
    long_notional: float = 0.0
    short_notional: float = 0.0
    net_notional: float = 0.0


class Decisions(BaseModel):
    per_ticker: dict[Ticker, TickerDecision] = Field(default_factory=dict)
    hedge_plan: HedgePlan = Field(default_factory=HedgePlan)


# ── Final output ────────────────────────────────────────────


class Recommendation(BaseModel):
    """Top-level run output. What `nova-v2` prints / serializes."""

    run_id: str
    as_of: str
    tickers: list[str]
    signals: list[Signal]
    consensus: dict[Ticker, Consensus] = Field(default_factory=dict)
    limits: Limits = Field(default_factory=Limits)
    decisions: Decisions = Field(default_factory=Decisions)
    summary: str = ""
    # Explain-only council narratives — how the risk + portfolio managers reasoned
    # over all the analyst signals. Verdict-immutable; empty when /reasoning is off.
    risk_reasoning: str = ""
    portfolio_reasoning: str = ""
