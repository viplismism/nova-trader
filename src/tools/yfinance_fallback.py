"""Yahoo Finance fallback — free data source when financialdatasets.ai is unavailable.

Used automatically by api.py when the primary API returns no data (no API key or 404).
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from src.data.models import (
    Price,
    FinancialMetrics,
    CompanyNews,
    InsiderTrade,
)


def is_available() -> bool:
    return _YF_AVAILABLE


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch daily OHLCV from Yahoo Finance."""
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            return []
        prices = []
        for dt, row in df.iterrows():
            prices.append(Price(
                open=float(row["Open"]),
                close=float(row["Close"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                volume=int(row["Volume"]),
                time=dt.strftime("%Y-%m-%dT00:00:00Z"),
            ))
        return prices
    except Exception as e:
        logger.warning("yfinance get_prices failed for %s: %s", ticker, e)
        return []


def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
    """Build FinancialMetrics from Yahoo Finance .info and quarterly financials."""
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        # Build a single TTM metrics record from info
        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=end_date,
            period=period,
            currency=info.get("currency", "USD"),
            market_cap=info.get("marketCap"),
            enterprise_value=info.get("enterpriseValue"),
            price_to_earnings_ratio=info.get("trailingPE") or info.get("forwardPE"),
            price_to_book_ratio=info.get("priceToBook"),
            price_to_sales_ratio=info.get("priceToSalesTrailing12Months"),
            enterprise_value_to_ebitda_ratio=info.get("enterpriseToEbitda"),
            enterprise_value_to_revenue_ratio=info.get("enterpriseToRevenue"),
            free_cash_flow_yield=_safe_div(info.get("freeCashflow"), info.get("marketCap")),
            peg_ratio=info.get("pegRatio"),
            gross_margin=info.get("grossMargins"),
            operating_margin=info.get("operatingMargins"),
            net_margin=info.get("profitMargins"),
            return_on_equity=info.get("returnOnEquity"),
            return_on_assets=info.get("returnOnAssets"),
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=_safe_div(info.get("debtToEquity"), 100),  # yf returns as %
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            book_value_growth=None,
            earnings_per_share_growth=info.get("earningsQuarterlyGrowth"),
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=info.get("payoutRatio"),
            earnings_per_share=info.get("trailingEps"),
            book_value_per_share=info.get("bookValue"),
            free_cash_flow_per_share=_safe_div(info.get("freeCashflow"), info.get("sharesOutstanding")),
        )
        return [metrics]
    except Exception as e:
        logger.warning("yfinance get_financial_metrics failed for %s: %s", ticker, e)
        return []


def get_market_cap(ticker: str) -> float | None:
    """Get market cap from Yahoo Finance."""
    if not _YF_AVAILABLE:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info.get("marketCap")
    except Exception as e:
        logger.warning("yfinance get_market_cap failed for %s: %s", ticker, e)
        return None


def get_company_news(ticker: str, limit: int = 20) -> list[CompanyNews]:
    """Fetch recent news from Yahoo Finance."""
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
        results = []
        for item in news_items[:limit]:
            content = item.get("content", {})
            pub_date = content.get("pubDate", "")
            # Convert to YYYY-MM-DD format if possible
            try:
                if pub_date:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    pub_date = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                pass
            provider = content.get("provider", {})
            results.append(CompanyNews(
                ticker=ticker,
                title=content.get("title", item.get("title", "")),
                author=None,
                source=provider.get("displayName", "Yahoo Finance"),
                date=pub_date,
                url=content.get("canonicalUrl", {}).get("url", ""),
            ))
        return results
    except Exception as e:
        logger.warning("yfinance get_company_news failed for %s: %s", ticker, e)
        return []


def get_insider_trades(ticker: str, limit: int = 50) -> list[InsiderTrade]:
    """Fetch insider transactions from Yahoo Finance."""
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        df = t.insider_transactions
        if df is None or df.empty:
            return []
        trades = []
        for _, row in df.head(limit).iterrows():
            start_date = ""
            if hasattr(row.get("Start Date", ""), "strftime"):
                start_date = row["Start Date"].strftime("%Y-%m-%d")
            elif isinstance(row.get("Start Date"), str):
                start_date = row["Start Date"]

            trades.append(InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=str(row.get("Insider", "")),
                title=str(row.get("Position", "")),
                is_board_director=None,
                transaction_date=start_date or None,
                transaction_shares=_safe_float(row.get("Shares")),
                transaction_price_per_share=None,
                transaction_value=_safe_float(row.get("Value")),
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=str(row.get("Text", "")),
                filing_date=start_date or datetime.now().strftime("%Y-%m-%d"),
            ))
        return trades
    except Exception as e:
        logger.warning("yfinance get_insider_trades failed for %s: %s", ticker, e)
        return []


# ── Helpers ───────────────────────────────────────────────

def _safe_div(a, b):
    """Safe division, returns None if either operand is None or b is 0."""
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def _safe_float(val):
    """Convert to float or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
