"""Fundamental signal scoring."""

from __future__ import annotations

from src.data.api import get_financial_metrics


def score_fundamentals(ticker: str, end_date: str, api_key: str | None = None) -> dict:
    metrics_list = get_financial_metrics(
        ticker=ticker,
        end_date=end_date,
        period="ttm",
        limit=10,
        api_key=api_key,
    )
    if not metrics_list:
        return {
            "signal": "neutral",
            "confidence": 0,
            "reasoning": "No financial metrics available",
            "status": "degraded",
        }

    m = metrics_list[0]
    signals: list[str] = []
    reasoning: dict[str, dict[str, str]] = {}

    prof_score = sum(
        v is not None and v > t
        for v, t in [
            (m.return_on_equity, 0.15),
            (m.net_margin, 0.20),
            (m.operating_margin, 0.15),
        ]
    )
    sig = "bullish" if prof_score >= 2 else "bearish" if prof_score == 0 else "neutral"
    signals.append(sig)
    reasoning["profitability_signal"] = {
        "signal": sig,
        "details": (
            (f"ROE: {m.return_on_equity:.2%}" if m.return_on_equity else "ROE: N/A")
            + ", "
            + (f"Net Margin: {m.net_margin:.2%}" if m.net_margin else "Net Margin: N/A")
            + ", "
            + (f"Op Margin: {m.operating_margin:.2%}" if m.operating_margin else "Op Margin: N/A")
        ),
    }

    growth_score = sum(
        v is not None and v > 0.10
        for v in [m.revenue_growth, m.earnings_growth, m.book_value_growth]
    )
    sig = "bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral"
    signals.append(sig)
    reasoning["growth_signal"] = {
        "signal": sig,
        "details": (
            (f"Revenue Growth: {m.revenue_growth:.2%}" if m.revenue_growth else "Revenue Growth: N/A")
            + ", "
            + (f"Earnings Growth: {m.earnings_growth:.2%}" if m.earnings_growth else "Earnings Growth: N/A")
        ),
    }

    health_score = 0
    if m.current_ratio and m.current_ratio > 1.5:
        health_score += 1
    if m.debt_to_equity and m.debt_to_equity < 0.5:
        health_score += 1
    if (
        m.free_cash_flow_per_share
        and m.earnings_per_share
        and m.free_cash_flow_per_share > m.earnings_per_share * 0.8
    ):
        health_score += 1
    sig = "bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral"
    signals.append(sig)
    reasoning["financial_health_signal"] = {
        "signal": sig,
        "details": (
            (f"Current Ratio: {m.current_ratio:.2f}" if m.current_ratio else "Current Ratio: N/A")
            + ", "
            + (f"D/E: {m.debt_to_equity:.2f}" if m.debt_to_equity else "D/E: N/A")
        ),
    }

    expensive_count = sum(
        v is not None and v > t
        for v, t in [
            (m.price_to_earnings_ratio, 25),
            (m.price_to_book_ratio, 3),
            (m.price_to_sales_ratio, 5),
        ]
    )
    sig = "bearish" if expensive_count >= 2 else "bullish" if expensive_count == 0 else "neutral"
    signals.append(sig)
    reasoning["price_ratios_signal"] = {
        "signal": sig,
        "details": (
            (f"P/E: {m.price_to_earnings_ratio:.2f}" if m.price_to_earnings_ratio else "P/E: N/A")
            + ", "
            + (f"P/B: {m.price_to_book_ratio:.2f}" if m.price_to_book_ratio else "P/B: N/A")
            + ", "
            + (f"P/S: {m.price_to_sales_ratio:.2f}" if m.price_to_sales_ratio else "P/S: N/A")
        ),
    }

    bull = signals.count("bullish")
    bear = signals.count("bearish")
    overall = "bullish" if bull > bear else "bearish" if bear > bull else "neutral"
    confidence = round(max(bull, bear) / len(signals), 2) * 100

    return {
        "signal": overall,
        "confidence": confidence,
        "reasoning": reasoning,
        "identity": "fundamentals_analyst",
    }
