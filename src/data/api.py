"""Market data access and normalization layer.

This module is the main entrypoint for prices, financial metrics, insider
trades, company news, and related lookup data. It handles HTTP requests,
caching, and Yahoo Finance fallbacks so the rest of the system can work with a
consistent data shape.
"""

import datetime
import logging
import os
import urllib.parse
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, params: dict = None, max_retries: int = 3) -> requests.Response:
    """
    Make an API request with rate limiting handling and moderate backoff.

    Args:
        url: The URL to request
        headers: Headers to include in the request
        method: HTTP method (GET or POST)
        json_data: JSON data for POST requests
        params: Query parameters dict (passed to requests as params=)
        max_retries: Maximum number of retries (default: 3)

    Returns:
        requests.Response: The response object

    Raises:
        Exception: If the request fails with a non-429 error
    """
    for attempt in range(max_retries + 1):  # +1 for initial attempt
        if method.upper() == "POST":
            response = requests.post(url, headers=headers, json=json_data, params=params, timeout=30)
        else:
            response = requests.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue
        
        # Return the response (whether success, other errors, or final 429)
        return response


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
    """Fetch price data from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date}_{end_date}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/prices/"
    response = _make_api_request(
        url, headers,
        params={"ticker": ticker, "interval": "day", "interval_multiplier": "1", "start_date": start_date, "end_date": end_date},
    )
    if response.status_code != 200:
        # Fallback to Yahoo Finance
        prices = _yf_get_prices(ticker, start_date, end_date)
        if prices:
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices

    # Parse response with Pydantic model
    try:
        price_response = PriceResponse(**response.json())
        prices = price_response.prices
    except Exception as e:
        logger.warning("Failed to parse price response for %s: %s", ticker, e)
        return _yf_get_prices(ticker, start_date, end_date)

    if not prices:
        # Fallback to Yahoo Finance
        prices = _yf_get_prices(ticker, start_date, end_date)
        if prices:
            _cache.set_prices(cache_key, [p.model_dump() for p in prices])
        return prices

    # Cache the results using the comprehensive cache key
    _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/financial-metrics/"
    response = _make_api_request(
        url, headers,
        params={"ticker": ticker, "report_period_lte": end_date, "limit": str(limit), "period": period},
    )
    if response.status_code != 200:
        # Fallback to Yahoo Finance
        metrics = _yf_get_financial_metrics(ticker, end_date, period, limit)
        if metrics:
            _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
        return metrics

    # Parse response with Pydantic model
    try:
        metrics_response = FinancialMetricsResponse(**response.json())
        financial_metrics = metrics_response.financial_metrics
    except Exception as e:
        logger.warning("Failed to parse financial metrics response for %s: %s", ticker, e)
        return _yf_get_financial_metrics(ticker, end_date, period, limit)

    if not financial_metrics:
        # Fallback to Yahoo Finance
        metrics = _yf_get_financial_metrics(ticker, end_date, period, limit)
        if metrics:
            _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
        return metrics

    # Cache the results as dicts using the comprehensive cache key
    _cache.set_financial_metrics(cache_key, [m.model_dump() for m in financial_metrics])
    return financial_metrics


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response.status_code != 200:
        return []
    
    try:
        data = response.json()
        response_model = LineItemResponse(**data)
        search_results = response_model.search_results
    except Exception as e:
        logger.warning("Failed to parse line items response for %s: %s", ticker, e)
        return []
    if not search_results:
        return []

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_trades = []
    current_end_date = end_date

    while True:
        params = {"ticker": ticker, "filing_date_lte": current_end_date, "limit": str(limit)}
        if start_date:
            params["filing_date_gte"] = start_date

        response = _make_api_request("https://api.financialdatasets.ai/insider-trades/", headers, params=params)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except Exception as e:
            logger.warning("Failed to parse insider trades response for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        # Fallback to Yahoo Finance
        trades = _yf_get_insider_trades(ticker, limit=limit)
        if trades:
            _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
        return trades

    # Cache the results using the comprehensive cache key
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_news = []
    current_end_date = end_date

    while True:
        params = {"ticker": ticker, "end_date": current_end_date, "limit": str(limit)}
        if start_date:
            params["start_date"] = start_date

        response = _make_api_request("https://api.financialdatasets.ai/news/", headers, params=params)
        if response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except Exception as e:
            logger.warning("Failed to parse company news response for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        # Fallback to Yahoo Finance
        news = _yf_get_company_news(ticker, limit=limit)
        if news:
            _cache.set_company_news(cache_key, [n.model_dump() for n in news])
        return news

    # Cache the results using the comprehensive cache key
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from the API."""
    # Check if end_date is today
    if end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        # Get the market cap from company facts API
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = "https://api.financialdatasets.ai/company/facts/"
        response = _make_api_request(url, headers, params={"ticker": ticker})
        if response.status_code != 200:
            # Fallback to Yahoo Finance
            return _yf_get_market_cap(ticker)

        data = response.json()
        response_model = CompanyFactsResponse(**data)
        return response_model.company_facts.market_cap

    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
    if not financial_metrics:
        return None

    market_cap = financial_metrics[0].market_cap

    if not market_cap:
        return None

    return market_cap


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)


# ---------------------------------------------------------------------------
# Yahoo Finance fallback helpers (used when primary API returns no data)
# ---------------------------------------------------------------------------

def _yf_safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return float(a) / float(b)


def _yf_safe_float(val):
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _yf_get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            return []
        return [
            Price(
                open=float(row["Open"]), close=float(row["Close"]),
                high=float(row["High"]), low=float(row["Low"]),
                volume=int(row["Volume"]),
                time=dt.strftime("%Y-%m-%dT00:00:00Z"),
            )
            for dt, row in df.iterrows()
        ]
    except Exception as e:
        logger.warning("yfinance get_prices failed for %s: %s", ticker, e)
        return []


def _yf_get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        metrics = FinancialMetrics(
            ticker=ticker, report_period=end_date, period=period,
            currency=info.get("currency", "USD"),
            market_cap=info.get("marketCap"),
            enterprise_value=info.get("enterpriseValue"),
            price_to_earnings_ratio=info.get("trailingPE") or info.get("forwardPE"),
            price_to_book_ratio=info.get("priceToBook"),
            price_to_sales_ratio=info.get("priceToSalesTrailing12Months"),
            enterprise_value_to_ebitda_ratio=info.get("enterpriseToEbitda"),
            enterprise_value_to_revenue_ratio=info.get("enterpriseToRevenue"),
            free_cash_flow_yield=_yf_safe_div(info.get("freeCashflow"), info.get("marketCap")),
            peg_ratio=info.get("pegRatio"),
            gross_margin=info.get("grossMargins"),
            operating_margin=info.get("operatingMargins"),
            net_margin=info.get("profitMargins"),
            return_on_equity=info.get("returnOnEquity"),
            return_on_assets=info.get("returnOnAssets"),
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            debt_to_equity=_yf_safe_div(info.get("debtToEquity"), 100),
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            earnings_per_share_growth=info.get("earningsQuarterlyGrowth"),
            payout_ratio=info.get("payoutRatio"),
            earnings_per_share=info.get("trailingEps"),
            book_value_per_share=info.get("bookValue"),
            free_cash_flow_per_share=_yf_safe_div(info.get("freeCashflow"), info.get("sharesOutstanding")),
        )
        return [metrics]
    except Exception as e:
        logger.warning("yfinance get_financial_metrics failed for %s: %s", ticker, e)
        return []


def _yf_get_market_cap(ticker: str) -> float | None:
    if not _YF_AVAILABLE:
        return None
    try:
        return (yf.Ticker(ticker).info or {}).get("marketCap")
    except Exception as e:
        logger.warning("yfinance get_market_cap failed for %s: %s", ticker, e)
        return None


def _yf_get_company_news(ticker: str, limit: int = 20) -> list[CompanyNews]:
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        results = []
        for item in (t.news or [])[:limit]:
            content = item.get("content", {})
            pub_date = content.get("pubDate", "")
            try:
                if pub_date:
                    dt = datetime.datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
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


def _yf_get_insider_trades(ticker: str, limit: int = 50) -> list[InsiderTrade]:
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
                ticker=ticker, issuer=None,
                name=str(row.get("Insider", "")),
                title=str(row.get("Position", "")),
                is_board_director=None,
                transaction_date=start_date or None,
                transaction_shares=_yf_safe_float(row.get("Shares")),
                transaction_price_per_share=None,
                transaction_value=_yf_safe_float(row.get("Value")),
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=None,
                security_title=str(row.get("Text", "")),
                filing_date=start_date or datetime.datetime.now().strftime("%Y-%m-%d"),
            ))
        return trades
    except Exception as e:
        logger.warning("yfinance get_insider_trades failed for %s: %s", ticker, e)
        return []
