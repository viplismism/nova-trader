"""Valuation analyst on the new contract.

Input:  FinancialsView (financial metrics + line items + market cap).
Output: Signal.

Combines four valuation methods (DCF, owner earnings, EV/EBITDA, residual
income) into a weighted gap vs market cap. Reuses the math from
src/agents/valuation.py.
"""

from __future__ import annotations

import logging

from src.agents.math.valuation import (
    calculate_cost_of_equity,
    calculate_dcf_scenarios,
    calculate_ev_ebitda_value,
    calculate_owner_earnings_value,
    calculate_residual_income_value,
    calculate_wacc,
)
from src.schemas.context import RunContext
from src.schemas.signals import Signal, ValuationTarget
from src.schemas.views import FinancialsView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "valuation"

METHOD_WEIGHTS = {
    "dcf": 0.35,
    "owner_earnings": 0.35,
    "ev_ebitda": 0.20,
    "residual_income": 0.10,
}


def run_valuation_agent(ctx: RunContext, view: FinancialsView, recorder=None) -> Signal:
    if not view.metrics or len(view.line_items) < 2:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker,
            reason="Need at least 2 periods of metrics and line items",
        )
    # Market cap drives every valuation gap. The dedicated live lookup is flaky
    # (no Financial Datasets key + rate-limited yfinance), so fall back to the
    # market_cap carried on the metrics we already fetched before giving up.
    market_cap = view.market_cap or getattr(view.metrics[0], "market_cap", None)
    if not market_cap:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker, reason="Market cap unavailable",
        )

    progress.update_status(AGENT_ID, view.ticker, "Running valuation models")

    try:
        most_recent = view.metrics[0]
        li_curr, li_prev = view.line_items[0], view.line_items[1]

        wc_change = 0
        if (getattr(li_curr, "working_capital", None) is not None and
                getattr(li_prev, "working_capital", None) is not None):
            wc_change = li_curr.working_capital - li_prev.working_capital

        owner_val = calculate_owner_earnings_value(
            net_income=getattr(li_curr, "net_income", None),
            depreciation=getattr(li_curr, "depreciation_and_amortization", None),
            capex=getattr(li_curr, "capital_expenditure", None),
            working_capital_change=wc_change,
            growth_rate=most_recent.earnings_growth or 0.05,
        )

        wacc = calculate_wacc(
            market_cap=most_recent.market_cap or view.market_cap or 0,
            total_debt=getattr(li_curr, "total_debt", None),
            cash=getattr(li_curr, "cash_and_equivalents", None),
            interest_coverage=most_recent.interest_coverage,
            debt_to_equity=most_recent.debt_to_equity,
        )

        fcf_history = [
            li.free_cash_flow
            for li in view.line_items
            if getattr(li, "free_cash_flow", None) is not None
        ]
        dcf_results = calculate_dcf_scenarios(
            fcf_history=fcf_history,
            growth_metrics={
                "revenue_growth": most_recent.revenue_growth,
                "fcf_growth": most_recent.free_cash_flow_growth,
                "earnings_growth": most_recent.earnings_growth,
            },
            wacc=wacc,
            market_cap=most_recent.market_cap or view.market_cap or 0,
            revenue_growth=most_recent.revenue_growth,
        )
        dcf_val = dcf_results.get("expected_value", 0)

        ev_ebitda_val = calculate_ev_ebitda_value(view.metrics)
        rim_val = calculate_residual_income_value(
            market_cap=most_recent.market_cap or view.market_cap,
            net_income=getattr(li_curr, "net_income", None),
            price_to_book_ratio=most_recent.price_to_book_ratio,
            book_value_growth=most_recent.book_value_growth or 0.03,
        )
    except Exception as e:
        logger.exception("valuation failed for %s", view.ticker)
        return Signal.failed(agent_id=AGENT_ID, ticker=view.ticker, error=str(e))

    method_values = {
        "dcf": dcf_val,
        "owner_earnings": owner_val,
        "ev_ebitda": ev_ebitda_val,
        "residual_income": rim_val,
    }

    contributing = {m: v for m, v in method_values.items() if v and v > 0}
    if not contributing:
        return Signal.abstained(
            agent_id=AGENT_ID, ticker=view.ticker, reason="All valuation methods returned zero",
        )

    total_weight = sum(METHOD_WEIGHTS[m] for m in contributing)
    weighted_gap = sum(
        METHOD_WEIGHTS[m] * (v - market_cap) / market_cap
        for m, v in contributing.items()
    ) / total_weight

    if weighted_gap > 0.15:
        direction = "bullish"
    elif weighted_gap < -0.15:
        direction = "bearish"
    else:
        direction = "neutral"
    confidence = min(0.95, abs(weighted_gap) / 0.30)

    key_factors = [
        f"{m}=${v:,.0f} (gap {(v - market_cap) / market_cap:+.1%})"
        for m, v in contributing.items()
    ]

    # 12-month price target: blended intrinsic value is market_cap * (1 + gap);
    # per share that is current_price * (1 + gap), then drifted forward one year
    # at the cost of equity. Needs shares outstanding to convert to a price.
    valuation_target = None
    shares = getattr(li_curr, "outstanding_shares", None)
    if shares and shares > 0:
        current_price = market_cap / shares
        fair_value = current_price * (1 + weighted_gap)
        cost_of_equity = calculate_cost_of_equity()
        target_price = fair_value * (1 + cost_of_equity)
        valuation_target = ValuationTarget(
            current_price=current_price,
            fair_value=fair_value,
            target_price=target_price,
            upside=target_price / current_price - 1,
            cost_of_equity=cost_of_equity,
        )
        key_factors.append(f"12mo target=${target_price:,.2f} ({valuation_target.upside:+.1%})")

    progress.update_status(AGENT_ID, view.ticker, "Done")

    return Signal(
        agent_id=AGENT_ID,
        ticker=view.ticker,
        direction=direction,
        confidence=confidence,
        reasoning=(
            f"Weighted valuation gap {weighted_gap:+.1%} vs market cap "
            f"${market_cap:,.0f} across {len(contributing)} methods"
        ),
        key_factors=key_factors,
        valuation_target=valuation_target,
    )
