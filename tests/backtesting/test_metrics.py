"""Unit tests for backtest performance metrics.

This module verifies the portfolio-value based metric calculations such as
Sharpe ratio, Sortino ratio, and drawdown under both normal and degenerate input
series.
"""

from datetime import datetime, timedelta

import numpy as np

from src.backtesting.metrics import PerformanceMetricsCalculator


def _build_values(values: list[float]):
    start = datetime(2024, 1, 1)
    points = []
    for i, v in enumerate(values):
        points.append({
            "Date": start + timedelta(days=i),
            "Portfolio Value": v,
            "Long Exposure": 0.0,
            "Short Exposure": 0.0,
            "Gross Exposure": 0.0,
            "Net Exposure": 0.0,
            "Long/Short Ratio": np.inf,
        })
    return points


def test_metrics_insufficient_data_returns_empty_values():
    calc = PerformanceMetricsCalculator()
    metrics = calc.compute_metrics(_build_values([100_000.0]))
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["max_drawdown"] is None


def test_metrics_basic_sharpe_sortino_and_drawdown():
    calc = PerformanceMetricsCalculator(annual_trading_days=2, annual_rf_rate=0.0)
    # Values: up then down → non-zero volatility; drawdown occurs on last day
    vals = _build_values([100.0, 110.0, 99.0])
    metrics = calc.compute_metrics(vals)
    assert metrics["sharpe_ratio"] is not None
    assert metrics["sortino_ratio"] is not None
    assert metrics["max_drawdown"] < 0.0
    assert isinstance(metrics.get("max_drawdown_date"), str)


def test_metrics_zero_volatility_sharpe_zero():
    calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
    # Constant portfolio value → zero volatility → Sharpe 0
    vals = _build_values([100.0, 100.0, 100.0, 100.0])
    metrics = calc.compute_metrics(vals)
    assert metrics["sharpe_ratio"] == 0.0
