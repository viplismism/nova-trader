"""Backtesting package for portfolio simulation utilities.

The current runtime uses these components through ``TradingService.backtest``.
This package intentionally keeps only the accounting, execution, valuation, cost,
and metric helpers needed by the lean core.
"""

from .types import (
    ActionLiteral,
    PerformanceMetrics,
    PortfolioSnapshot,
    PortfolioValuePoint,
    PositionState,
    TickerRealizedGains,
)

from .portfolio import Portfolio
from .trader import TradeExecutor
from .metrics import PerformanceMetricsCalculator
from .valuation import calculate_portfolio_value, compute_exposures

__all__ = [
    # Types
    "ActionLiteral",
    "PerformanceMetrics",
    "PortfolioSnapshot",
    "PortfolioValuePoint",
    "PositionState",
    "TickerRealizedGains",
    # Interfaces
    "Portfolio",
    "TradeExecutor",
    "PerformanceMetricsCalculator",
    "calculate_portfolio_value",
    "compute_exposures",
]
