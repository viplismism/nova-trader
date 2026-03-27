"""Reusable scoring functions for agent analysis.

Each function takes financial data and returns {"score": float, "max_score": float, "details": str}.
Agents compose these via their YAML configs.
"""

from __future__ import annotations


def _safe_attr(item, attr, default=None):
    """Safely get an attribute from a line item (may be a Pydantic model with extra fields)."""
    return getattr(item, attr, default)


def _extract_series(items: list, attr: str) -> list[float]:
    """Extract a numeric series from line items, skipping None."""
    return [v for item in items if (v := _safe_attr(item, attr)) is not None]


# ──────────────────────────────────────────────
# Profitability & Returns
# ──────────────────────────────────────────────

def score_roic(metrics: list, financial_line_items: list, threshold: float = 0.15, periods_pct: float = 0.8) -> dict:
    """Score Return on Invested Capital consistency above threshold."""
    roic_values = _extract_series(financial_line_items, "return_on_invested_capital")
    if not roic_values:
        # Fallback to metrics
        roic_values = [m.return_on_invested_capital for m in metrics if m.return_on_invested_capital is not None]

    if not roic_values:
        return {"score": 0, "max_score": 3, "details": "No ROIC data available"}

    high_count = sum(1 for r in roic_values if r > threshold)
    ratio = high_count / len(roic_values)

    if ratio >= periods_pct:
        return {"score": 3, "max_score": 3, "details": f"Excellent ROIC: >{threshold:.0%} in {high_count}/{len(roic_values)} periods"}
    elif ratio >= 0.5:
        return {"score": 2, "max_score": 3, "details": f"Good ROIC: >{threshold:.0%} in {high_count}/{len(roic_values)} periods"}
    elif high_count > 0:
        return {"score": 1, "max_score": 3, "details": f"Mixed ROIC: >{threshold:.0%} in {high_count}/{len(roic_values)} periods"}
    return {"score": 0, "max_score": 3, "details": f"Poor ROIC: never exceeds {threshold:.0%}"}


def score_roe(metrics: list, threshold: float = 0.15) -> dict:
    """Score Return on Equity consistency."""
    roe_values = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if not roe_values:
        return {"score": 0, "max_score": 2, "details": "No ROE data"}

    high_count = sum(1 for r in roe_values if r > threshold)
    if high_count >= len(roe_values) * 0.7:
        return {"score": 2, "max_score": 2, "details": f"Strong ROE: >{threshold:.0%} in {high_count}/{len(roe_values)} periods"}
    elif high_count > 0:
        return {"score": 1, "max_score": 2, "details": f"Mixed ROE: >{threshold:.0%} in {high_count}/{len(roe_values)} periods"}
    return {"score": 0, "max_score": 2, "details": f"Weak ROE: never exceeds {threshold:.0%}"}


def score_margins(metrics: list) -> dict:
    """Score gross/operating/net margin consistency."""
    details = []
    score = 0
    max_score = 3

    for margin_name, attr in [("Gross", "gross_margin"), ("Operating", "operating_margin"), ("Net", "net_margin")]:
        values = [getattr(m, attr) for m in metrics if getattr(m, attr, None) is not None]
        if values:
            avg = sum(values) / len(values)
            if avg > 0.20:
                score += 1
                details.append(f"{margin_name} margin {avg:.1%} (strong)")
            else:
                details.append(f"{margin_name} margin {avg:.1%}")

    return {"score": score, "max_score": max_score, "details": "; ".join(details) or "No margin data"}


# ──────────────────────────────────────────────
# Moat & Competitive Position
# ──────────────────────────────────────────────

def score_gross_margin_stability(financial_line_items: list) -> dict:
    """Score pricing power via gross margin trend."""
    margins = _extract_series(financial_line_items, "gross_margin")
    if len(margins) < 3:
        return {"score": 0, "max_score": 2, "details": "Insufficient gross margin data"}

    improving = sum(1 for i in range(1, len(margins)) if margins[i] >= margins[i - 1])
    if improving >= len(margins) * 0.7:
        return {"score": 2, "max_score": 2, "details": "Strong pricing power: margins consistently improving"}
    avg = sum(margins) / len(margins)
    if avg > 0.30:
        return {"score": 1, "max_score": 2, "details": f"Good pricing power: avg gross margin {avg:.1%}"}
    return {"score": 0, "max_score": 2, "details": "Limited pricing power"}


def score_capital_intensity(financial_line_items: list) -> dict:
    """Score low capex-to-revenue (asset-light business)."""
    ratios = []
    for item in financial_line_items:
        capex = _safe_attr(item, "capital_expenditure")
        rev = _safe_attr(item, "revenue")
        if capex is not None and rev and rev > 0:
            ratios.append(abs(capex) / rev)

    if not ratios:
        return {"score": 0, "max_score": 2, "details": "No capex data"}

    avg = sum(ratios) / len(ratios)
    if avg < 0.05:
        return {"score": 2, "max_score": 2, "details": f"Asset-light: avg capex {avg:.1%} of revenue"}
    elif avg < 0.10:
        return {"score": 1, "max_score": 2, "details": f"Moderate capex: {avg:.1%} of revenue"}
    return {"score": 0, "max_score": 2, "details": f"Capital-intensive: {avg:.1%} of revenue"}


# ──────────────────────────────────────────────
# Financial Strength
# ──────────────────────────────────────────────

def score_debt_to_equity(metrics: list, max_safe: float = 0.5) -> dict:
    """Score leverage safety."""
    de_values = [m.debt_to_equity for m in metrics if m.debt_to_equity is not None]
    if not de_values:
        return {"score": 0, "max_score": 2, "details": "No D/E data"}

    latest = de_values[0]
    if latest < max_safe * 0.6:
        return {"score": 2, "max_score": 2, "details": f"Conservative leverage: D/E {latest:.2f}"}
    elif latest < max_safe:
        return {"score": 1, "max_score": 2, "details": f"Moderate leverage: D/E {latest:.2f}"}
    return {"score": 0, "max_score": 2, "details": f"High leverage: D/E {latest:.2f}"}


def score_current_ratio(metrics: list) -> dict:
    """Score liquidity."""
    cr_values = [m.current_ratio for m in metrics if m.current_ratio is not None]
    if not cr_values:
        return {"score": 0, "max_score": 1, "details": "No current ratio data"}

    latest = cr_values[0]
    if latest >= 2.0:
        return {"score": 1, "max_score": 1, "details": f"Strong liquidity: CR {latest:.2f}"}
    return {"score": 0, "max_score": 1, "details": f"Tight liquidity: CR {latest:.2f}"}


def score_fcf_conversion(financial_line_items: list) -> dict:
    """Score free cash flow to net income conversion."""
    fcf = _extract_series(financial_line_items, "free_cash_flow")
    ni = _extract_series(financial_line_items, "net_income")

    if not fcf or not ni or len(fcf) != len(ni):
        return {"score": 0, "max_score": 2, "details": "Missing FCF or NI data"}

    ratios = [f / n for f, n in zip(fcf, ni) if n and n > 0]
    if not ratios:
        return {"score": 0, "max_score": 2, "details": "Cannot compute FCF/NI ratio"}

    avg = sum(ratios) / len(ratios)
    if avg > 1.0:
        return {"score": 2, "max_score": 2, "details": f"Excellent cash conversion: FCF/NI {avg:.2f}"}
    elif avg > 0.8:
        return {"score": 1, "max_score": 2, "details": f"Good cash conversion: FCF/NI {avg:.2f}"}
    return {"score": 0, "max_score": 2, "details": f"Poor cash conversion: FCF/NI {avg:.2f}"}


# ──────────────────────────────────────────────
# Growth & Consistency
# ──────────────────────────────────────────────

def score_revenue_consistency(financial_line_items: list, min_periods: int = 5) -> dict:
    """Score revenue growth consistency."""
    revs = _extract_series(financial_line_items, "revenue")
    if len(revs) < min_periods:
        return {"score": 0, "max_score": 3, "details": f"Need {min_periods}+ periods, have {len(revs)}"}

    growth_rates = []
    for i in range(len(revs) - 1):
        if revs[i + 1] and revs[i + 1] != 0:
            growth_rates.append(revs[i] / revs[i + 1] - 1)

    if not growth_rates:
        return {"score": 0, "max_score": 3, "details": "Cannot compute revenue growth"}

    avg = sum(growth_rates) / len(growth_rates)
    vol = sum(abs(r - avg) for r in growth_rates) / len(growth_rates)

    if avg > 0.05 and vol < 0.10:
        return {"score": 3, "max_score": 3, "details": f"Highly predictable revenue: {avg:.1%} avg growth, low volatility"}
    elif avg > 0 and vol < 0.20:
        return {"score": 2, "max_score": 3, "details": f"Moderate revenue stability: {avg:.1%} avg growth"}
    elif avg > 0:
        return {"score": 1, "max_score": 3, "details": f"Growing but volatile revenue: {avg:.1%} growth"}
    return {"score": 0, "max_score": 3, "details": f"Declining revenue: {avg:.1%} growth"}


def score_earnings_stability(financial_line_items: list) -> dict:
    """Score earnings (EPS/net income) consistency."""
    eps = _extract_series(financial_line_items, "earnings_per_share")
    if not eps:
        ni = _extract_series(financial_line_items, "net_income")
        eps = ni  # Fallback to net income

    if len(eps) < 3:
        return {"score": 0, "max_score": 2, "details": "Insufficient earnings data"}

    positive = sum(1 for e in eps if e > 0)
    if positive == len(eps):
        return {"score": 2, "max_score": 2, "details": f"Consistently profitable: {positive}/{len(eps)} positive periods"}
    elif positive >= len(eps) * 0.7:
        return {"score": 1, "max_score": 2, "details": f"Mostly profitable: {positive}/{len(eps)} positive periods"}
    return {"score": 0, "max_score": 2, "details": f"Erratic earnings: only {positive}/{len(eps)} profitable periods"}


def score_book_value_growth(financial_line_items: list) -> dict:
    """Score book value per share growth."""
    bv = _extract_series(financial_line_items, "book_value_per_share")
    if not bv:
        # Fallback: shareholders_equity / outstanding_shares
        equity = _extract_series(financial_line_items, "shareholders_equity")
        shares = _extract_series(financial_line_items, "outstanding_shares")
        if equity and shares and len(equity) == len(shares):
            bv = [e / s for e, s in zip(equity, shares) if s and s > 0]

    if len(bv) < 3:
        return {"score": 0, "max_score": 2, "details": "Insufficient book value data"}

    growing = sum(1 for i in range(len(bv) - 1) if bv[i] >= bv[i + 1])
    if growing >= (len(bv) - 1) * 0.7:
        return {"score": 2, "max_score": 2, "details": "Book value consistently growing"}
    elif growing >= (len(bv) - 1) * 0.5:
        return {"score": 1, "max_score": 2, "details": "Book value moderately growing"}
    return {"score": 0, "max_score": 2, "details": "Book value stagnant or declining"}


# ──────────────────────────────────────────────
# Management Quality
# ──────────────────────────────────────────────

def score_share_dilution(financial_line_items: list) -> dict:
    """Score share count trend (buybacks vs dilution)."""
    shares = _extract_series(financial_line_items, "outstanding_shares")
    if len(shares) < 3:
        return {"score": 0, "max_score": 2, "details": "Insufficient share count data"}

    first, last = shares[0], shares[-1]
    if first < last * 0.95:
        return {"score": 2, "max_score": 2, "details": "Reducing share count (buybacks)"}
    elif first < last * 1.05:
        return {"score": 1, "max_score": 2, "details": "Stable share count"}
    return {"score": 0, "max_score": 2, "details": "Significant share dilution"}


def score_insider_activity(insider_trades: list) -> dict:
    """Score insider buy vs sell activity."""
    if not insider_trades:
        return {"score": 0, "max_score": 2, "details": "No insider trading data"}

    # Use transaction_shares: negative = sell, positive = buy
    buys = sum(1 for t in insider_trades if (ts := _safe_attr(t, "transaction_shares")) is not None and ts > 0)
    sells = sum(1 for t in insider_trades if (ts := _safe_attr(t, "transaction_shares")) is not None and ts < 0)
    total = buys + sells

    if total == 0:
        return {"score": 0, "max_score": 2, "details": "No buy/sell transactions"}

    buy_ratio = buys / total
    if buy_ratio > 0.6:
        return {"score": 2, "max_score": 2, "details": f"Strong insider buying: {buys}/{total} buys"}
    elif buy_ratio > 0.3:
        return {"score": 1, "max_score": 2, "details": f"Mixed insider activity: {buys}/{total} buys"}
    return {"score": 0, "max_score": 2, "details": f"Heavy insider selling: {sells}/{total} sells"}


# ──────────────────────────────────────────────
# Valuation
# ──────────────────────────────────────────────

def score_intrinsic_value(financial_line_items: list, market_cap: float | None, growth_rate: float = 0.05, discount_rate: float = 0.10, terminal_multiple: float = 15) -> dict:
    """Simple DCF-based intrinsic value vs market cap."""
    fcf = _extract_series(financial_line_items, "free_cash_flow")
    if not fcf or not market_cap or market_cap <= 0:
        return {"score": 0, "max_score": 3, "details": "Missing FCF or market cap for valuation", "intrinsic_value": None, "margin_of_safety": None}

    base_fcf = fcf[0]
    if base_fcf <= 0:
        return {"score": 0, "max_score": 3, "details": f"Negative FCF ({base_fcf:,.0f}), can't value", "intrinsic_value": None, "margin_of_safety": None}

    # 5-year projection + terminal
    pv = 0
    for year in range(1, 6):
        projected = base_fcf * (1 + growth_rate) ** year
        pv += projected / (1 + discount_rate) ** year

    terminal = (base_fcf * (1 + growth_rate) ** 5 * terminal_multiple) / (1 + discount_rate) ** 5
    intrinsic = pv + terminal
    margin = (intrinsic - market_cap) / market_cap

    if margin > 0.30:
        return {"score": 3, "max_score": 3, "details": f"Undervalued: {margin:.0%} margin of safety", "intrinsic_value": intrinsic, "margin_of_safety": margin}
    elif margin > 0:
        return {"score": 2, "max_score": 3, "details": f"Fairly valued: {margin:.0%} margin", "intrinsic_value": intrinsic, "margin_of_safety": margin}
    elif margin > -0.20:
        return {"score": 1, "max_score": 3, "details": f"Slightly overvalued: {margin:.0%}", "intrinsic_value": intrinsic, "margin_of_safety": margin}
    return {"score": 0, "max_score": 3, "details": f"Overvalued: {margin:.0%}", "intrinsic_value": intrinsic, "margin_of_safety": margin}


def score_pe_ratio(metrics: list, max_pe: float = 25) -> dict:
    """Score P/E relative to a threshold."""
    pe_values = [m.price_to_earnings_ratio for m in metrics if m.price_to_earnings_ratio is not None]
    if not pe_values:
        return {"score": 0, "max_score": 2, "details": "No P/E data"}

    latest = pe_values[0]
    if latest < 0:
        return {"score": 0, "max_score": 2, "details": f"Negative P/E ({latest:.1f}) — company losing money"}
    if latest < max_pe * 0.6:
        return {"score": 2, "max_score": 2, "details": f"Cheap: P/E {latest:.1f} (threshold: {max_pe})"}
    elif latest < max_pe:
        return {"score": 1, "max_score": 2, "details": f"Fair P/E: {latest:.1f}"}
    return {"score": 0, "max_score": 2, "details": f"Expensive: P/E {latest:.1f} (threshold: {max_pe})"}


def score_news_sentiment(company_news: list) -> dict:
    """Quick sentiment tally from pre-tagged news."""
    if not company_news:
        return {"score": 0, "max_score": 1, "details": "No news data"}

    positive = sum(1 for n in company_news if getattr(n, "sentiment", None) == "positive")
    negative = sum(1 for n in company_news if getattr(n, "sentiment", None) == "negative")

    if positive > negative * 2:
        return {"score": 1, "max_score": 1, "details": f"Positive news sentiment ({positive} pos, {negative} neg)"}
    elif negative > positive * 2:
        return {"score": 0, "max_score": 1, "details": f"Negative news sentiment ({positive} pos, {negative} neg)"}
    return {"score": 0, "max_score": 1, "details": f"Mixed news ({positive} pos, {negative} neg)"}


# ──────────────────────────────────────────────
# Registry: name → function mapping
# ──────────────────────────────────────────────

SCORING_FUNCTIONS = {
    "roic": score_roic,
    "roe": score_roe,
    "margins": score_margins,
    "gross_margin_stability": score_gross_margin_stability,
    "capital_intensity": score_capital_intensity,
    "debt_to_equity": score_debt_to_equity,
    "current_ratio": score_current_ratio,
    "fcf_conversion": score_fcf_conversion,
    "revenue_consistency": score_revenue_consistency,
    "earnings_stability": score_earnings_stability,
    "book_value_growth": score_book_value_growth,
    "share_dilution": score_share_dilution,
    "insider_activity": score_insider_activity,
    "intrinsic_value": score_intrinsic_value,
    "pe_ratio": score_pe_ratio,
    "news_sentiment": score_news_sentiment,
}
