"""Typed contracts for Nova Trader's recommendation pipeline.

Every cross-component handoff goes through one of these schemas.
No dict-of-dict state, no fake chat messages — just typed objects.
"""

from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio, Position, RealizedGains
from src.schemas.signals import (
    Consensus,
    Decisions,
    HedgePair,
    HedgePlan,
    Limits,
    Recommendation,
    Signal,
    SignalStatus,
    TickerDecision,
    TickerLimit,
)
from src.schemas.snapshot import MarketSnapshot
from src.schemas.views import (
    FinancialsView,
    InsiderView,
    NewsSentimentView,
    PersonaView,
    PortfolioView,
    PriceView,
)

__all__ = [
    "ModelConfig",
    "RunContext",
    "RunRequest",
    "Portfolio",
    "Position",
    "RealizedGains",
    "Signal",
    "SignalStatus",
    "Consensus",
    "Limits",
    "TickerLimit",
    "Decisions",
    "TickerDecision",
    "HedgePair",
    "HedgePlan",
    "Recommendation",
    "MarketSnapshot",
    "PriceView",
    "FinancialsView",
    "PersonaView",
    "NewsSentimentView",
    "InsiderView",
    "PortfolioView",
]
