"""Per-agent input views.

Each view is the EXACT contract for what an agent can see.
Pydantic enforces the contract — an agent that asks for fields not in
its view simply won't compile.

The mapping agent_id -> ViewClass lives in src/v2/registry.py.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.data.models import (
    CompanyNews,
    FilingExcerpt,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
    RedditPost,
    WebSearchResult,
)
from src.schemas.portfolio import Portfolio
from src.schemas.signals import Consensus

Ticker = str


# ── Single-ticker views (passed per ticker) ─────────────────


class PriceView(BaseModel):
    """For the technical agent. Prices only."""

    ticker: str
    prices: list[Price]

    model_config = {"arbitrary_types_allowed": True}


class FinancialsView(BaseModel):
    """For quant agents (fundamentals, growth, valuation)."""

    ticker: str
    metrics: list[FinancialMetrics]
    line_items: list[LineItem]
    market_cap: float | None

    model_config = {"arbitrary_types_allowed": True}


class PersonaView(BaseModel):
    """For every investor persona (Buffett, Munger, Ackman, ...).

    Per the design conversation: all personas get financials + news + insider.
    The persona's prompt template decides what to weight.
    """

    ticker: str
    metrics: list[FinancialMetrics]
    line_items: list[LineItem]
    market_cap: float | None
    news: list[CompanyNews]
    insider: list[InsiderTrade]

    model_config = {"arbitrary_types_allowed": True}


class NewsSentimentView(BaseModel):
    """For the news sentiment agent. News + prices to read reaction."""

    ticker: str
    news: list[CompanyNews]
    prices: list[Price]

    model_config = {"arbitrary_types_allowed": True}


class InsiderView(BaseModel):
    """For the insider sentiment agent."""

    ticker: str
    trades: list[InsiderTrade]

    model_config = {"arbitrary_types_allowed": True}


class FilingsView(BaseModel):
    """For the SEC filings analyst."""

    ticker: str
    excerpts: list[FilingExcerpt]

    model_config = {"arbitrary_types_allowed": True}


class WebResearchView(BaseModel):
    """For the live web research analyst."""

    ticker: str
    results: list[WebSearchResult]

    model_config = {"arbitrary_types_allowed": True}


class RedditView(BaseModel):
    """For the Reddit / retail-sentiment analyst."""

    ticker: str
    posts: list[RedditPost]

    model_config = {"arbitrary_types_allowed": True}


class AdaptiveResearchView(BaseModel):
    """For the tool-use-style adaptive research analyst."""

    ticker: str
    metrics: list[FinancialMetrics]
    market_cap: float | None
    news: list[CompanyNews]
    filings: list[FilingExcerpt]
    web_results: list[WebSearchResult]

    model_config = {"arbitrary_types_allowed": True}


# ── Multi-ticker views (passed once, not per ticker) ────────


class PortfolioView(BaseModel):
    """For risk + portfolio managers.

    Sees the whole portfolio, the consensus across all tickers, and price
    history (needed for volatility and correlation math). Does NOT see
    individual agent signals — only the aggregated consensus.
    """

    portfolio: Portfolio
    consensus: dict[Ticker, Consensus] = Field(default_factory=dict)
    prices: dict[Ticker, list[Price]] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
