"""MarketSnapshot — the single source of data for one run.

Built once at the top of the pipeline. Every agent receives a *slice*
of this snapshot via a typed view, never the snapshot directly.
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

Ticker = str


class MarketSnapshot(BaseModel):
    """All raw data needed for a run, fetched once.

    Stored per-ticker. Agents do not call APIs themselves — the snapshot
    is the only data source. This guarantees every agent sees the same
    facts and gives us a single artifact to log per run.
    """

    prices: dict[Ticker, list[Price]] = Field(default_factory=dict)
    financials: dict[Ticker, list[FinancialMetrics]] = Field(default_factory=dict)
    line_items: dict[Ticker, list[LineItem]] = Field(default_factory=dict)
    news: dict[Ticker, list[CompanyNews]] = Field(default_factory=dict)
    insider: dict[Ticker, list[InsiderTrade]] = Field(default_factory=dict)
    filings: dict[Ticker, list[FilingExcerpt]] = Field(default_factory=dict)
    web_research: dict[Ticker, list[WebSearchResult]] = Field(default_factory=dict)
    reddit: dict[Ticker, list[RedditPost]] = Field(default_factory=dict)
    market_cap: dict[Ticker, float | None] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}
