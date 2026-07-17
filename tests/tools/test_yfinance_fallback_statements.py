"""Multi-period statement fallback — the fix for growth/valuation abstaining on
every ticker the primary data API doesn't cover (they need 4 / 2 periods)."""

import pandas as pd
import pytest

from src.tools import yfinance_fallback as yfb


def _frame(rows: dict, periods: list[str]) -> pd.DataFrame:
    cols = [pd.Timestamp(p) for p in periods]
    return pd.DataFrame({c: {k: v[i] for k, v in rows.items()} for i, c in enumerate(cols)})


_PERIODS = ["2025-08-31", "2024-08-31", "2023-08-31", "2022-08-31"]


class _StubTicker:
    income_stmt = _frame({
        "Total Revenue": [37e9, 25e9, 15e9, 30e9],
        "Net Income": [8e9, 0.7e9, -5e9, 8.6e9],
        "Gross Profit": [14e9, 5e9, -1e9, 13e9],
        "Operating Income": [9e9, 1e9, -5.7e9, 9.7e9],
    }, _PERIODS)
    balance_sheet = _frame({
        "Total Assets": [80e9, 69e9, 64e9, 66e9],
        "Total Liabilities Net Minority Interest": [30e9, 24e9, 20e9, 16e9],
        "Stockholders Equity": [50e9, 45e9, 44e9, 50e9],
        "Ordinary Shares Number": [1.12e9, 1.11e9, 1.09e9, 1.1e9],
        "Working Capital": [15e9, 13e9, 12e9, 14e9],
        "Total Debt": [14e9, 13e9, 13e9, 7e9],
        "Cash And Cash Equivalents": [8e9, 7e9, 8.5e9, 8e9],
        "Current Assets": [25e9, 20e9, 18e9, 22e9],
        "Current Liabilities": [10e9, 9e9, 8e9, 9e9],
    }, _PERIODS)
    cashflow = _frame({
        "Free Cash Flow": [1.6e9, -3e9, -6e9, 3.1e9],
        "Operating Cash Flow": [17e9, 8e9, 1.5e9, 15e9],
        "Capital Expenditure": [-15.4e9, -11e9, -7.5e9, -11.9e9],
        "Depreciation And Amortization": [8e9, 7.7e9, 7.6e9, 7e9],
        "Cash Dividends Paid": [-0.5e9, -0.5e9, -0.5e9, -0.4e9],
        "Repurchase Of Capital Stock": [-0.1e9, 0, -0.4e9, -2.4e9],
    }, _PERIODS)
    info = {"currency": "USD", "marketCap": 963e9, "trailingPE": 20.4, "priceToBook": 19.0}


@pytest.fixture(autouse=True)
def _stub_yf(monkeypatch):
    yfb._cache_clear()
    monkeypatch.setattr(yfb.yf, "Ticker", lambda _t: _StubTicker())
    yield
    yfb._cache_clear()


def test_line_items_multi_period_with_capex_normalized():
    li = yfb.search_line_items("MU", [], "2026-07-15")
    assert len(li) == 4
    assert li[0].report_period == "2025-08-31"  # newest first
    assert li[0].capital_expenditure == pytest.approx(15.4e9)  # positive spend
    assert li[0].free_cash_flow == pytest.approx(1.6e9)
    assert li[0].outstanding_shares == pytest.approx(1.12e9)
    assert li[0].working_capital == pytest.approx(15e9)


def test_metrics_multi_period_with_growth_and_overlay():
    m = yfb.get_financial_metrics("MU", "2026-07-15")
    assert len(m) == 4
    assert m[0].revenue_growth == pytest.approx(37 / 25 - 1)
    assert m[0].net_margin == pytest.approx(8 / 37, rel=1e-3)
    assert m[0].current_ratio == pytest.approx(2.5)
    assert m[0].market_cap == pytest.approx(963e9)     # info overlay on newest
    assert m[1].market_cap is None                      # not on older periods
    assert m[3].revenue_growth is None                  # oldest has no prior year


def test_growth_sign_safe_across_loss_to_profit():
    # 2023 net income was -5e9, 2022 was +8.6e9: 2023 "growth" must be NEGATIVE
    # (profit -> loss) and 2024 (loss -> small profit) must be POSITIVE — the
    # naive cur/prev-1 inverts both when prev < 0.
    m = yfb.get_financial_metrics("MU", "2026-07-15")
    g_2024 = m[1].earnings_growth  # 0.7e9 vs -5e9: improvement
    g_2023 = m[2].earnings_growth  # -5e9 vs 8.6e9: collapse
    assert g_2024 is not None and g_2024 > 0
    assert g_2023 is not None and g_2023 < 0


def test_statements_fetched_once_per_ticker(monkeypatch):
    calls = {"n": 0}
    class _Counting(_StubTicker):
        def __init__(self):
            calls["n"] += 1
    yfb._cache_clear()
    monkeypatch.setattr(yfb.yf, "Ticker", lambda _t: _Counting())
    yfb.get_financial_metrics("MU", "2026-07-15")
    yfb.search_line_items("MU", [], "2026-07-15")
    # one Ticker for .info + one inside _statement_rows; the second function
    # must hit the statements cache instead of re-downloading
    assert calls["n"] <= 2
