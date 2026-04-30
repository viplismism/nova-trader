"""Tests for the deterministic factor investing scorer.

These cases validate that the factor model combines value, momentum, quality,
and related sleeves into a stable signal and degrades gracefully when required
data is missing.
"""

from datetime import date, timedelta

from src.alpha.features.factor import score_factor_investing
from src.data.models import FinancialMetrics, Price


def _prices_from_closes(closes: list[float]) -> list[Price]:
    prices: list[Price] = []
    start = date(2025, 1, 1)
    for idx, close in enumerate(closes, start=1):
        ts = (start + timedelta(days=idx - 1)).isoformat()
        prices.append(
            Price(
                open=close,
                close=close,
                high=close * 1.01,
                low=close * 0.99,
                volume=1_000_000,
                time=f"{ts}T00:00:00Z",
            )
        )
    return prices


def test_factor_investing_bullish_multifactor(monkeypatch):
    metrics = FinancialMetrics(
        ticker="AAPL",
        report_period="2025-12-31",
        period="ttm",
        currency="USD",
        market_cap=1_000_000_000,
        enterprise_value=1_100_000_000,
        price_to_earnings_ratio=15.0,
        price_to_book_ratio=2.0,
        price_to_sales_ratio=3.0,
        enterprise_value_to_ebitda_ratio=10.0,
        enterprise_value_to_revenue_ratio=4.0,
        free_cash_flow_yield=0.06,
        peg_ratio=1.0,
        gross_margin=0.52,
        operating_margin=0.24,
        net_margin=0.19,
        return_on_equity=0.22,
        return_on_assets=0.10,
        return_on_invested_capital=0.16,
        asset_turnover=1.1,
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=1.8,
        quick_ratio=1.5,
        cash_ratio=0.7,
        operating_cash_flow_ratio=0.25,
        debt_to_equity=0.35,
        debt_to_assets=0.20,
        interest_coverage=8.0,
        revenue_growth=0.12,
        earnings_growth=0.15,
        book_value_growth=0.09,
        earnings_per_share_growth=0.14,
        free_cash_flow_growth=0.11,
        operating_income_growth=0.12,
        ebitda_growth=0.12,
        payout_ratio=0.20,
        earnings_per_share=6.0,
        book_value_per_share=30.0,
        free_cash_flow_per_share=5.5,
    )

    closes = [100 + (i * 0.5) for i in range(140)]

    monkeypatch.setattr(
        "src.alpha.features.factor.get_financial_metrics",
        lambda *args, **kwargs: [metrics],
    )
    monkeypatch.setattr(
        "src.alpha.features.factor.get_prices",
        lambda *args, **kwargs: _prices_from_closes(closes),
    )

    result = score_factor_investing("AAPL", "2026-03-31")

    assert result["signal"] == "bullish"
    assert result["status"] == "success"
    assert result["confidence"] > 50
    assert result["reasoning"]["value"]["signal"] == "bullish"
    assert result["reasoning"]["quality"]["signal"] == "bullish"
    assert result["reasoning"]["multi_factor"]["signal"] == "bullish"


def test_factor_investing_degraded_when_prices_missing(monkeypatch):
    metrics = FinancialMetrics(
        ticker="MSFT",
        report_period="2025-12-31",
        period="ttm",
        currency="USD",
        market_cap=1_000_000_000,
        enterprise_value=1_050_000_000,
        price_to_earnings_ratio=18.0,
        price_to_book_ratio=2.5,
        price_to_sales_ratio=3.5,
        enterprise_value_to_ebitda_ratio=11.0,
        enterprise_value_to_revenue_ratio=4.5,
        free_cash_flow_yield=0.05,
        peg_ratio=1.2,
        gross_margin=0.50,
        operating_margin=0.21,
        net_margin=0.17,
        return_on_equity=0.20,
        return_on_assets=0.09,
        return_on_invested_capital=0.15,
        asset_turnover=1.0,
        inventory_turnover=None,
        receivables_turnover=None,
        days_sales_outstanding=None,
        operating_cycle=None,
        working_capital_turnover=None,
        current_ratio=1.6,
        quick_ratio=1.3,
        cash_ratio=0.6,
        operating_cash_flow_ratio=0.21,
        debt_to_equity=0.40,
        debt_to_assets=0.22,
        interest_coverage=7.0,
        revenue_growth=0.11,
        earnings_growth=0.12,
        book_value_growth=0.08,
        earnings_per_share_growth=0.10,
        free_cash_flow_growth=0.09,
        operating_income_growth=0.10,
        ebitda_growth=0.10,
        payout_ratio=0.22,
        earnings_per_share=5.5,
        book_value_per_share=28.0,
        free_cash_flow_per_share=5.0,
    )

    monkeypatch.setattr(
        "src.alpha.features.factor.get_financial_metrics",
        lambda *args, **kwargs: [metrics],
    )
    monkeypatch.setattr(
        "src.alpha.features.factor.get_prices",
        lambda *args, **kwargs: [],
    )

    result = score_factor_investing("MSFT", "2026-03-31")

    assert result["status"] == "degraded"
    assert result["signal"] in {"bullish", "neutral"}
    assert "Missing price history" in result["error"]
