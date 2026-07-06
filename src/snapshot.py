"""Build a MarketSnapshot once at the start of a run.

Agents never call data APIs themselves. They get a slice of the snapshot
via a typed view. One source of truth per run, one set of API calls.
"""

from __future__ import annotations

import logging

from src.schemas.context import RunContext
from src.schemas.snapshot import MarketSnapshot
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    search_line_items,
)
from src.tools.community import get_community_posts
from src.tools.reddit import get_reddit_posts
from src.tools.sec_filings import get_sec_filing_excerpts
from src.tools.web_search import get_web_research
from src.utils.progress import current_fetch_owner, progress

logger = logging.getLogger(__name__)

# Comprehensive line-item set covering Buffett-style + fundamentals + valuation.
# Fetched once per ticker so every persona/quant agent can read from the same data.
DEFAULT_LINE_ITEMS = [
    "revenue",
    "net_income",
    "gross_profit",
    "operating_income",
    "free_cash_flow",
    "capital_expenditure",
    "depreciation_and_amortization",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "outstanding_shares",
    "dividends_and_other_cash_distributions",
    "issuance_or_purchase_of_equity_shares",
]


def build_snapshot(ctx: RunContext, api_key: str | None = None) -> MarketSnapshot:
    """Fetch all market data for the run, once.

    Errors per-ticker are logged and skipped — the snapshot for that ticker
    will be partial. Agents downstream see the missing fields and can
    abstain/fail appropriately.
    """
    snapshot = MarketSnapshot()

    # All fetches in this phase are attributed to "snapshot" (analysts read slices,
    # they never fetch) so the run-total fetch badge counts them under one owner.
    owner_token = current_fetch_owner.set("snapshot")
    try:
        _build_snapshot_body(ctx, snapshot, api_key)
    finally:
        current_fetch_owner.reset(owner_token)
    return snapshot


def _build_snapshot_body(ctx: RunContext, snapshot: MarketSnapshot, api_key: str | None) -> None:
    selected = set(ctx.request.selected_agents or [])
    fetch_filings = not selected or "sec_filings" in selected
    fetch_web = not selected or "web_research" in selected
    fetch_reddit = not selected or "reddit_sentiment" in selected
    for ticker in ctx.tickers:
        progress.update_status("snapshot", ticker, "Fetching prices")
        try:
            snapshot.prices[ticker] = get_prices(
                ticker=ticker,
                start_date=ctx.start_date,
                end_date=ctx.end_date,
                api_key=api_key,
            ) or []
        except Exception as e:
            logger.warning("snapshot prices failed for %s: %s", ticker, e)
            snapshot.prices[ticker] = []

        progress.update_status("snapshot", ticker, "Fetching financials")
        try:
            snapshot.financials[ticker] = get_financial_metrics(
                ticker=ticker,
                end_date=ctx.end_date,
                period="ttm",
                limit=10,
                api_key=api_key,
            ) or []
        except Exception as e:
            logger.warning("snapshot financials failed for %s: %s", ticker, e)
            snapshot.financials[ticker] = []

        progress.update_status("snapshot", ticker, "Fetching line items")
        try:
            snapshot.line_items[ticker] = search_line_items(
                ticker=ticker,
                line_items=DEFAULT_LINE_ITEMS,
                end_date=ctx.end_date,
                period="ttm",
                limit=10,
                api_key=api_key,
            ) or []
        except Exception as e:
            logger.warning("snapshot line_items failed for %s: %s", ticker, e)
            snapshot.line_items[ticker] = []

        progress.update_status("snapshot", ticker, "Fetching market cap")
        try:
            snapshot.market_cap[ticker] = get_market_cap(
                ticker=ticker,
                end_date=ctx.end_date,
                api_key=api_key,
            )
        except Exception as e:
            logger.warning("snapshot market_cap failed for %s: %s", ticker, e)
            snapshot.market_cap[ticker] = None  # type: ignore[assignment]

        progress.update_status("snapshot", ticker, "Fetching news")
        try:
            snapshot.news[ticker] = get_company_news(
                ticker=ticker,
                end_date=ctx.end_date,
                limit=25,
                api_key=api_key,
            ) or []
        except Exception as e:
            logger.warning("snapshot news failed for %s: %s", ticker, e)
            snapshot.news[ticker] = []

        progress.update_status("snapshot", ticker, "Fetching insider trades")
        try:
            snapshot.insider[ticker] = get_insider_trades(
                ticker=ticker,
                end_date=ctx.end_date,
                limit=100,
                api_key=api_key,
            ) or []
        except Exception as e:
            logger.warning("snapshot insider failed for %s: %s", ticker, e)
            snapshot.insider[ticker] = []

        if fetch_filings:
            progress.update_status("snapshot", ticker, "Fetching SEC filings")
            try:
                snapshot.filings[ticker] = get_sec_filing_excerpts(ticker=ticker) or []
            except Exception as e:
                logger.warning("snapshot SEC filings failed for %s: %s", ticker, e)
                snapshot.filings[ticker] = []
        else:
            snapshot.filings[ticker] = []

        if fetch_web:
            progress.update_status("snapshot", ticker, "Searching the web")
            try:
                snapshot.web_research[ticker] = get_web_research(ticker=ticker) or []
            except Exception as e:
                logger.warning("snapshot web research failed for %s: %s", ticker, e)
                snapshot.web_research[ticker] = []
        else:
            snapshot.web_research[ticker] = []

        if fetch_reddit:
            progress.update_status("snapshot", ticker, "Reading Reddit")
            try:
                snapshot.reddit[ticker] = get_reddit_posts(ticker=ticker) or []
            except Exception as e:
                logger.warning("snapshot reddit failed for %s: %s", ticker, e)
                snapshot.reddit[ticker] = []
            progress.update_status("snapshot", ticker, "Reading community feed")
            try:
                snapshot.community[ticker] = [p.text for p in get_community_posts(ticker) or []]
            except Exception as e:
                logger.warning("snapshot community feed failed for %s: %s", ticker, e)
                snapshot.community[ticker] = []
        else:
            snapshot.reddit[ticker] = []
            snapshot.community[ticker] = []

        progress.update_status("snapshot", ticker, "Done")
