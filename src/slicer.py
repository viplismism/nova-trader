"""Build typed views from the snapshot.

This is where the "each agent sees only what its contract allows" rule
is enforced. The slicer is the one place that knows how a given view
class is filled from the snapshot.

Adding a new view: add a builder here + register it.
"""

from __future__ import annotations

from typing import Callable, Type

from pydantic import BaseModel

from src.schemas.signals import Consensus
from src.schemas.snapshot import MarketSnapshot
from src.schemas.portfolio import Portfolio
from src.schemas.views import (
    AdaptiveResearchView,
    FilingsView,
    RedditView,
    FinancialsView,
    InsiderView,
    NewsSentimentView,
    PersonaView,
    PortfolioView,
    PriceView,
    WebResearchView,
)


# ── Per-ticker view builders ────────────────────────────────


def _build_price_view(snapshot: MarketSnapshot, ticker: str) -> PriceView:
    return PriceView(
        ticker=ticker,
        prices=snapshot.prices.get(ticker, []),
    )


def _build_financials_view(snapshot: MarketSnapshot, ticker: str) -> FinancialsView:
    return FinancialsView(
        ticker=ticker,
        metrics=snapshot.financials.get(ticker, []),
        line_items=snapshot.line_items.get(ticker, []),
        market_cap=snapshot.market_cap.get(ticker),
    )


def _build_persona_view(snapshot: MarketSnapshot, ticker: str) -> PersonaView:
    return PersonaView(
        ticker=ticker,
        metrics=snapshot.financials.get(ticker, []),
        line_items=snapshot.line_items.get(ticker, []),
        market_cap=snapshot.market_cap.get(ticker),
        news=snapshot.news.get(ticker, []),
        insider=snapshot.insider.get(ticker, []),
    )


def _build_news_sentiment_view(snapshot: MarketSnapshot, ticker: str) -> NewsSentimentView:
    return NewsSentimentView(
        ticker=ticker,
        news=snapshot.news.get(ticker, []),
        prices=snapshot.prices.get(ticker, []),
    )


def _build_insider_view(snapshot: MarketSnapshot, ticker: str) -> InsiderView:
    return InsiderView(
        ticker=ticker,
        trades=snapshot.insider.get(ticker, []),
    )


def _build_filings_view(snapshot: MarketSnapshot, ticker: str) -> FilingsView:
    return FilingsView(
        ticker=ticker,
        excerpts=snapshot.filings.get(ticker, []),
    )


def _build_web_research_view(snapshot: MarketSnapshot, ticker: str) -> WebResearchView:
    return WebResearchView(
        ticker=ticker,
        results=snapshot.web_research.get(ticker, []),
    )


def _build_reddit_view(snapshot: MarketSnapshot, ticker: str) -> "RedditView":
    return RedditView(
        ticker=ticker,
        posts=snapshot.reddit.get(ticker, []),
    )


def _build_adaptive_research_view(snapshot: MarketSnapshot, ticker: str) -> AdaptiveResearchView:
    return AdaptiveResearchView(
        ticker=ticker,
        metrics=snapshot.financials.get(ticker, []),
        market_cap=snapshot.market_cap.get(ticker),
        news=snapshot.news.get(ticker, []),
        filings=snapshot.filings.get(ticker, []),
        web_results=snapshot.web_research.get(ticker, []),
    )


PER_TICKER_BUILDERS: dict[Type[BaseModel], Callable[[MarketSnapshot, str], BaseModel]] = {
    AdaptiveResearchView: _build_adaptive_research_view,
    PriceView: _build_price_view,
    FinancialsView: _build_financials_view,
    PersonaView: _build_persona_view,
    NewsSentimentView: _build_news_sentiment_view,
    InsiderView: _build_insider_view,
    FilingsView: _build_filings_view,
    WebResearchView: _build_web_research_view,
    RedditView: _build_reddit_view,
}


def build_view(view_class: Type[BaseModel], snapshot: MarketSnapshot, ticker: str) -> BaseModel:
    """Build a per-ticker view of the requested class."""
    builder = PER_TICKER_BUILDERS.get(view_class)
    if builder is None:
        raise ValueError(f"No view builder registered for {view_class.__name__}")
    return builder(snapshot, ticker)


# ── Portfolio-level view builder ────────────────────────────


def build_portfolio_view(
    snapshot: MarketSnapshot,
    portfolio: Portfolio,
    consensus: dict[str, Consensus],
) -> PortfolioView:
    """Build the view passed to risk + portfolio managers.

    They see the portfolio, the consensus across all tickers, and prices
    (for vol + correlation math). They do NOT see individual agent signals.
    """
    return PortfolioView(
        portfolio=portfolio,
        consensus=consensus,
        prices=snapshot.prices,
    )
