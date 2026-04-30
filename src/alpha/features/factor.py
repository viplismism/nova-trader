"""Factor-investing signal scoring.

Inspired by AQR's high-level factor framing:
value, momentum, quality, low volatility, defensive, and multi-factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import sqrt

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.core.signals import SignalResult
from src.data.api import get_financial_metrics, get_prices, prices_to_df


@dataclass
class FactorLeg:
    name: str
    score: float
    signal: str
    details: dict[str, float | int | str | None]


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _signal_from_score(score: float, bullish: float = 0.20, bearish: float = -0.20) -> str:
    if score >= bullish:
        return "bullish"
    if score <= bearish:
        return "bearish"
    return "neutral"


def _safe_metric_pairs(pairs: list[tuple[float | None, float, str]]) -> tuple[int, int, int]:
    available = 0
    bullish = 0
    bearish = 0
    for value, threshold, mode in pairs:
        if value is None:
            continue
        available += 1
        if mode == "low":
            if value <= threshold:
                bullish += 1
            else:
                bearish += 1
        else:
            if value >= threshold:
                bullish += 1
            else:
                bearish += 1
    return available, bullish, bearish


def _value_leg(metrics) -> FactorLeg:
    available, cheap, expensive = _safe_metric_pairs(
        [
            (metrics.price_to_earnings_ratio, 20.0, "low"),
            (metrics.price_to_book_ratio, 3.0, "low"),
            (metrics.price_to_sales_ratio, 4.0, "low"),
            (metrics.peg_ratio, 1.5, "low"),
        ]
    )

    if metrics.free_cash_flow_yield is not None:
        available += 1
        if metrics.free_cash_flow_yield >= 0.04:
            cheap += 1
        elif metrics.free_cash_flow_yield <= 0.01:
            expensive += 1

    score = 0.0 if available == 0 else _clamp((cheap - expensive) / available)
    return FactorLeg(
        name="value",
        score=score,
        signal=_signal_from_score(score),
        details={
            "cheap_signals": cheap,
            "expensive_signals": expensive,
            "metrics_used": available,
            "pe": metrics.price_to_earnings_ratio,
            "pb": metrics.price_to_book_ratio,
            "ps": metrics.price_to_sales_ratio,
            "fcf_yield": metrics.free_cash_flow_yield,
            "peg": metrics.peg_ratio,
        },
    )


def _quality_leg(metrics) -> FactorLeg:
    available, strong, weak = _safe_metric_pairs(
        [
            (metrics.return_on_equity, 0.15, "high"),
            (metrics.return_on_assets, 0.07, "high"),
            (metrics.return_on_invested_capital, 0.10, "high"),
            (metrics.gross_margin, 0.40, "high"),
            (metrics.net_margin, 0.10, "high"),
            (metrics.current_ratio, 1.20, "high"),
            (metrics.interest_coverage, 4.0, "high"),
        ]
    )

    if metrics.debt_to_equity is not None:
        available += 1
        if metrics.debt_to_equity <= 0.75:
            strong += 1
        elif metrics.debt_to_equity >= 1.5:
            weak += 1

    score = 0.0 if available == 0 else _clamp((strong - weak) / available)
    return FactorLeg(
        name="quality",
        score=score,
        signal=_signal_from_score(score),
        details={
            "strong_signals": strong,
            "weak_signals": weak,
            "metrics_used": available,
            "roe": metrics.return_on_equity,
            "roa": metrics.return_on_assets,
            "roic": metrics.return_on_invested_capital,
            "gross_margin": metrics.gross_margin,
            "net_margin": metrics.net_margin,
            "current_ratio": metrics.current_ratio,
            "debt_to_equity": metrics.debt_to_equity,
        },
    )


def _momentum_leg(prices_df: pd.DataFrame) -> FactorLeg:
    close = prices_df["close"]
    if len(close) < 126:
        return FactorLeg(
            name="momentum",
            score=0.0,
            signal="neutral",
            details={"error": "Insufficient price history for 6M momentum", "observations": len(close)},
        )

    mom_1m = float(close.iloc[-1] / close.iloc[-22] - 1.0) if len(close) >= 22 else 0.0
    mom_3m = float(close.iloc[-1] / close.iloc[-64] - 1.0) if len(close) >= 64 else 0.0
    mom_6m = float(close.iloc[-1] / close.iloc[-127] - 1.0) if len(close) >= 127 else 0.0
    weighted = (0.30 * mom_1m) + (0.30 * mom_3m) + (0.40 * mom_6m)

    score = _clamp(weighted / 0.20)
    return FactorLeg(
        name="momentum",
        score=score,
        signal=_signal_from_score(score),
        details={
            "return_1m": round(mom_1m, 4),
            "return_3m": round(mom_3m, 4),
            "return_6m": round(mom_6m, 4),
            "weighted_return": round(weighted, 4),
            "observations": len(close),
        },
    )


def _low_vol_leg(prices_df: pd.DataFrame) -> FactorLeg:
    close = prices_df["close"]
    returns = close.pct_change().dropna()
    if len(returns) < 20:
        return FactorLeg(
            name="low_volatility",
            score=0.0,
            signal="neutral",
            details={"error": "Insufficient return history", "observations": len(returns)},
        )

    ann_vol = float(returns.std() * sqrt(252))
    drawdown = float((close / close.cummax() - 1.0).min())

    if ann_vol <= 0.20 and drawdown >= -0.20:
        score = 0.8
    elif ann_vol <= 0.28 and drawdown >= -0.30:
        score = 0.3
    elif ann_vol >= 0.45 or drawdown <= -0.45:
        score = -0.8
    elif ann_vol >= 0.35 or drawdown <= -0.35:
        score = -0.3
    else:
        score = 0.0

    return FactorLeg(
        name="low_volatility",
        score=score,
        signal=_signal_from_score(score),
        details={
            "annualized_volatility": round(ann_vol, 4),
            "max_drawdown": round(drawdown, 4),
            "observations": len(returns),
        },
    )


def _composite_leg(name: str, legs: list[tuple[FactorLeg, float]]) -> FactorLeg:
    total_weight = sum(weight for _, weight in legs)
    if total_weight <= 0:
        score = 0.0
    else:
        score = _clamp(sum(leg.score * weight for leg, weight in legs) / total_weight)
    return FactorLeg(
        name=name,
        score=score,
        signal=_signal_from_score(score),
        details={
            "components": {
                leg.name: {
                    "signal": leg.signal,
                    "score": round(leg.score, 4),
                    "weight": weight,
                }
                for leg, weight in legs
            }
        },
    )


def score_factor_investing(ticker: str, end_date: str, api_key: str | None = None) -> dict:
    """Score a ticker using a simple multi-factor framework."""
    metrics_list = get_financial_metrics(
        ticker=ticker,
        end_date=end_date,
        period="ttm",
        limit=10,
        api_key=api_key,
    )

    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_date = (end_dt - relativedelta(years=1)).strftime("%Y-%m-%d")
    prices = get_prices(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        api_key=api_key,
    )
    prices_df = prices_to_df(prices) if prices else pd.DataFrame()

    if not metrics_list and prices_df.empty:
        return SignalResult.failed(
            error="No financial metrics or price history available",
            agent_id="factor_investing_analyst",
        ).to_dict()

    status = "success"
    errors: list[str] = []

    if not metrics_list:
        status = "degraded"
        errors.append("Missing financial metrics")
    if prices_df.empty:
        status = "degraded"
        errors.append("Missing price history")

    legs: dict[str, FactorLeg] = {}

    if metrics_list:
        metrics = metrics_list[0]
        legs["value"] = _value_leg(metrics)
        legs["quality"] = _quality_leg(metrics)
    else:
        legs["value"] = FactorLeg("value", 0.0, "neutral", {"error": "Missing financial metrics"})
        legs["quality"] = FactorLeg("quality", 0.0, "neutral", {"error": "Missing financial metrics"})

    if not prices_df.empty:
        legs["momentum"] = _momentum_leg(prices_df)
        legs["low_volatility"] = _low_vol_leg(prices_df)
    else:
        legs["momentum"] = FactorLeg("momentum", 0.0, "neutral", {"error": "Missing price history"})
        legs["low_volatility"] = FactorLeg("low_volatility", 0.0, "neutral", {"error": "Missing price history"})

    legs["defensive"] = _composite_leg(
        "defensive",
        [(legs["quality"], 0.60), (legs["low_volatility"], 0.40)],
    )
    legs["multi_factor"] = _composite_leg(
        "multi_factor",
        [(legs["value"], 0.35), (legs["momentum"], 0.35), (legs["quality"], 0.30)],
    )

    overall_score = _clamp(
        (0.30 * legs["value"].score)
        + (0.25 * legs["momentum"].score)
        + (0.25 * legs["quality"].score)
        + (0.10 * legs["low_volatility"].score)
        + (0.10 * legs["defensive"].score)
    )
    overall_signal = _signal_from_score(overall_score)

    aligned = sum(1 for leg in legs.values() if leg.signal == overall_signal and leg.signal != "neutral")
    confidence = min(95.0, round(abs(overall_score) * 70 + (aligned * 5), 1))

    reasoning = {
        leg.name: {
            "signal": leg.signal,
            "score": round(leg.score, 4),
            "details": leg.details,
        }
        for leg in legs.values()
    }
    reasoning["overall"] = {
        "signal": overall_signal,
        "score": round(overall_score, 4),
        "aligned_legs": aligned,
        "summary": (
            f"Factor blend based on value, momentum, quality, low volatility, "
            f"and defensive sleeves produced a {overall_signal} view."
        ),
    }

    if status == "degraded":
        return SignalResult.degraded(
            signal=overall_signal,
            confidence=confidence,
            reasoning=reasoning,
            error="; ".join(errors),
            agent_id="factor_investing_analyst",
        ).to_dict()

    return SignalResult.success(
        signal=overall_signal,
        confidence=confidence,
        reasoning=reasoning,
        agent_id="factor_investing_analyst",
        data_sources=["financial_metrics", "prices"],
    ).to_dict()
