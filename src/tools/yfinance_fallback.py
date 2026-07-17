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
    LineItem,
)


def is_available() -> bool:
    return _YF_AVAILABLE


# ── Annual-statement plumbing ────────────────────────────────
# Yahoo's annual statements give ~4 periods of history — exactly what the
# growth (needs >=4) and valuation (needs >=2) analysts require. Row labels
# vary across yfinance versions, so each field lists its known aliases.


def _first_row(frame, *labels) -> dict:
    """First matching statement row as {period_timestamp: float}."""
    if frame is None or getattr(frame, "empty", True):
        return {}
    for label in labels:
        if label in frame.index:
            return {c: _safe_float(v) for c, v in frame.loc[label].items()}
    return {}


def _statement_rows(t) -> tuple[list, dict[str, dict]]:
    """All statement rows we consume, plus the union of period columns (newest first)."""
    inc, bal, cf = t.income_stmt, t.balance_sheet, t.cashflow
    rows = {
        "revenue": _first_row(inc, "Total Revenue", "Operating Revenue"),
        "net_income": _first_row(inc, "Net Income", "Net Income Common Stockholders"),
        "gross_profit": _first_row(inc, "Gross Profit"),
        "operating_income": _first_row(inc, "Operating Income", "Total Operating Income As Reported"),
        "free_cash_flow": _first_row(cf, "Free Cash Flow"),
        "operating_cash_flow": _first_row(cf, "Operating Cash Flow"),
        "capital_expenditure": _first_row(cf, "Capital Expenditure"),
        "depreciation_and_amortization": _first_row(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion"),
        "dividends_and_other_cash_distributions": _first_row(cf, "Cash Dividends Paid", "Common Stock Dividend Paid"),
        "issuance_or_purchase_of_equity_shares": _first_row(cf, "Repurchase Of Capital Stock", "Common Stock Issuance"),
        "total_assets": _first_row(bal, "Total Assets"),
        "total_liabilities": _first_row(bal, "Total Liabilities Net Minority Interest"),
        "shareholders_equity": _first_row(bal, "Stockholders Equity", "Common Stock Equity"),
        "outstanding_shares": _first_row(bal, "Ordinary Shares Number", "Share Issued"),
        "working_capital": _first_row(bal, "Working Capital"),
        "total_debt": _first_row(bal, "Total Debt"),
        "cash_and_equivalents": _first_row(bal, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
        "current_assets": _first_row(bal, "Current Assets"),
        "current_liabilities": _first_row(bal, "Current Liabilities"),
    }
    cols = sorted({c for r in rows.values() for c in r}, reverse=True)
    return cols, rows


def search_line_items(ticker: str, line_items: list[str], end_date: str,
                      period: str = "ttm", limit: int = 10) -> list[LineItem]:
    """Build multi-period LineItems from Yahoo's annual statements (newest first)."""
    if not _YF_AVAILABLE:
        return []
    try:
        cols, rows = _statement_rows(yf.Ticker(ticker))
        out: list[LineItem] = []
        for col in cols[:limit]:
            fields = {name: row.get(col) for name, row in rows.items()}
            # Yahoo reports capex as a negative cash outflow; the valuation math
            # (owner earnings = NI + D&A - capex - ΔWC) expects positive spend.
            if fields.get("capital_expenditure") is not None:
                fields["capital_expenditure"] = abs(fields["capital_expenditure"])
            # Derive FCF when the direct row is missing: op cash flow + capex(neg).
            if fields.get("free_cash_flow") is None and fields.get("operating_cash_flow") is not None:
                capex_neg = -(fields.get("capital_expenditure") or 0)
                fields["free_cash_flow"] = fields["operating_cash_flow"] + capex_neg
            out.append(LineItem(
                ticker=ticker,
                report_period=col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col),
                period=period,
                currency="USD",
                **fields,
            ))
        return out
    except Exception as e:
        logger.warning("yfinance search_line_items failed for %s: %s", ticker, e)
        return []


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


def _blank_metrics(ticker: str, report_period: str, period: str, currency: str) -> dict:
    fields = {name: None for name in FinancialMetrics.model_fields}
    fields.update(ticker=ticker, report_period=report_period, period=period, currency=currency)
    return fields


def _info_overlay(info: dict) -> dict:
    """Price-dependent ratios only .info can provide — applied to the newest period."""
    overlay = {
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "price_to_earnings_ratio": info.get("trailingPE") or info.get("forwardPE"),
        "price_to_book_ratio": info.get("priceToBook"),
        "price_to_sales_ratio": info.get("priceToSalesTrailing12Months"),
        "enterprise_value_to_ebitda_ratio": info.get("enterpriseToEbitda"),
        "enterprise_value_to_revenue_ratio": info.get("enterpriseToRevenue"),
        "free_cash_flow_yield": _safe_div(info.get("freeCashflow"), info.get("marketCap")),
        "peg_ratio": info.get("pegRatio"),
        "return_on_assets": info.get("returnOnAssets"),
        "quick_ratio": info.get("quickRatio"),
        "payout_ratio": info.get("payoutRatio"),
    }
    return {k: v for k, v in overlay.items() if v is not None}


def get_financial_metrics(ticker: str, end_date: str, period: str = "ttm", limit: int = 10) -> list[FinancialMetrics]:
    """Build MULTI-PERIOD FinancialMetrics from Yahoo's annual statements.

    The growth analyst needs >=4 periods and the fundamentals math needs trends,
    so a single .info snapshot is not enough — statements give ~4 years. The
    newest period is enriched with .info's live valuation ratios (P/E etc.),
    which statements alone cannot supply. Falls back to the one-row .info
    record when statements are unavailable (e.g. some ETFs/foreign listings).
    """
    if not _YF_AVAILABLE:
        return []
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        currency = info.get("currency", "USD")
        cols, rows = _statement_rows(t)

        def g(name: str, i: int):
            return rows[name].get(cols[i]) if i < len(cols) else None

        def yoy(name: str, i: int):
            cur, prev = g(name, i), g(name, i + 1)
            if cur is None or not prev:
                return None
            return cur / prev - 1

        records: list[FinancialMetrics] = []
        for i, col in enumerate(cols[:limit]):
            rev, ni = g("revenue", i), g("net_income", i)
            equity, shares = g("shareholders_equity", i), g("outstanding_shares", i)
            eps_now = _safe_div(ni, shares)
            eps_prev = _safe_div(g("net_income", i + 1), g("outstanding_shares", i + 1))
            fields = _blank_metrics(
                ticker,
                col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col),
                period,
                currency,
            )
            fields.update(
                gross_margin=_safe_div(g("gross_profit", i), rev),
                operating_margin=_safe_div(g("operating_income", i), rev),
                net_margin=_safe_div(ni, rev),
                return_on_equity=_safe_div(ni, equity),
                debt_to_equity=_safe_div(g("total_debt", i), equity),
                current_ratio=_safe_div(g("current_assets", i), g("current_liabilities", i)),
                revenue_growth=yoy("revenue", i),
                earnings_growth=yoy("net_income", i),
                book_value_growth=yoy("shareholders_equity", i),
                free_cash_flow_growth=yoy("free_cash_flow", i),
                operating_income_growth=yoy("operating_income", i),
                earnings_per_share_growth=(eps_now / eps_prev - 1) if eps_now is not None and eps_prev else None,
                earnings_per_share=eps_now,
                book_value_per_share=_safe_div(equity, shares),
                free_cash_flow_per_share=_safe_div(g("free_cash_flow", i), shares),
            )
            records.append(FinancialMetrics(**fields))

        if not records:
            # Statements unavailable — degrade to the single .info snapshot.
            fields = _blank_metrics(ticker, end_date, period, currency)
            fields.update(
                gross_margin=info.get("grossMargins"),
                operating_margin=info.get("operatingMargins"),
                net_margin=info.get("profitMargins"),
                return_on_equity=info.get("returnOnEquity"),
                current_ratio=info.get("currentRatio"),
                debt_to_equity=_safe_div(info.get("debtToEquity"), 100),  # yf returns as %
                revenue_growth=info.get("revenueGrowth"),
                earnings_growth=info.get("earningsGrowth"),
                earnings_per_share_growth=info.get("earningsQuarterlyGrowth"),
                earnings_per_share=info.get("trailingEps"),
                book_value_per_share=info.get("bookValue"),
                free_cash_flow_per_share=_safe_div(info.get("freeCashflow"), info.get("sharesOutstanding")),
                interest_coverage=None,
            )
            fields.update(_info_overlay(info))
            return [FinancialMetrics(**fields)]

        # Live valuation ratios belong to today's price -> newest period only.
        records[0] = records[0].model_copy(update=_info_overlay(info))
        return records
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
