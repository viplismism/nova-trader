"""Risk manager on the new contract.

Input:  PortfolioView (portfolio + consensus + per-ticker prices).
Output: Limits (per-ticker volatility-adjusted position limits).

Math reused from src/agents/risk_manager.py.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.agents.math.risk_manager import (
    calculate_correlation_multiplier,
    calculate_volatility_adjusted_limit,
    calculate_volatility_metrics,
)
from src.schemas.context import RunContext
from src.schemas.signals import Limits, TickerLimit
from src.schemas.views import PortfolioView
from src.tools.api import prices_to_df
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "risk_manager"


def _portfolio_value(view: PortfolioView, current_prices: dict[str, float]) -> float:
    """Net liquidation value: cash + market value of longs - market value of shorts."""
    value = view.portfolio.cash
    for ticker, pos in view.portfolio.positions.items():
        price = current_prices.get(ticker, 0.0)
        value += pos.long * price
        value -= pos.short * price
    return value


def run_risk_manager(ctx: RunContext, view: PortfolioView) -> Limits:
    limits = Limits()

    # 1. Per-ticker volatility + current price
    vol_data: dict[str, dict] = {}
    current_prices: dict[str, float] = {}
    returns_by_ticker: dict[str, pd.Series] = {}

    all_tickers = set(ctx.tickers) | set(view.portfolio.positions.keys())
    for ticker in all_tickers:
        prices = view.prices.get(ticker, [])
        if not prices:
            progress.update_status(AGENT_ID, None, f"{ticker}: no price data")
            vol_data[ticker] = {
                "daily_volatility": 0.05,
                "annualized_volatility": 0.05 * (252 ** 0.5),
                "volatility_percentile": 100,
                "data_points": 0,
            }
            current_prices[ticker] = 0.0
            continue

        df = prices_to_df(prices)
        if df.empty or len(df) < 2:
            current_prices[ticker] = 0.0
            vol_data[ticker] = {
                "daily_volatility": 0.05,
                "annualized_volatility": 0.05 * (252 ** 0.5),
                "volatility_percentile": 100,
                "data_points": 0,
            }
            continue

        current_prices[ticker] = float(df["close"].iloc[-1])
        vol_data[ticker] = calculate_volatility_metrics(df)
        returns = df["close"].pct_change().dropna()
        if len(returns) > 0:
            returns_by_ticker[ticker] = returns
        progress.update_status(
            AGENT_ID, None,
            f"{ticker}: price {current_prices[ticker]:.2f}, vol {vol_data[ticker]['annualized_volatility']:.1%}",
        )

    # 2. Correlation matrix across active tickers
    correlation_matrix = None
    if len(returns_by_ticker) >= 2:
        try:
            returns_df = pd.DataFrame(returns_by_ticker).dropna(how="any")
            if returns_df.shape[1] >= 2 and returns_df.shape[0] >= 5:
                correlation_matrix = returns_df.corr()
        except Exception as e:
            logger.warning("correlation matrix failed: %s", e)

    active_positions = {
        t for t, pos in view.portfolio.positions.items()
        if abs(pos.long - pos.short) > 0
    }
    total_value = _portfolio_value(view, current_prices)
    progress.update_status(AGENT_ID, None, f"Total portfolio value: {total_value:.2f}")

    # 3. Per-ticker limits
    for ticker in ctx.tickers:
        price = current_prices.get(ticker, 0.0)
        if price <= 0:
            limits.per_ticker[ticker] = TickerLimit(
                ticker=ticker, current_price=0.0,
                max_position_dollars=0.0, max_shares=0,
                annualized_volatility=0.0, correlation_multiplier=1.0,
                remaining_position_limit=0.0,
            )
            continue

        annualized_vol = vol_data[ticker]["annualized_volatility"]
        base_pct = calculate_volatility_adjusted_limit(annualized_vol)

        corr_mult = 1.0
        if correlation_matrix is not None and ticker in correlation_matrix.columns:
            comparable = [t for t in active_positions if t in correlation_matrix.columns and t != ticker]
            if not comparable:
                comparable = [t for t in correlation_matrix.columns if t != ticker]
            if comparable:
                series = correlation_matrix.loc[ticker, comparable].dropna()
                if len(series) > 0:
                    corr_mult = calculate_correlation_multiplier(float(series.mean()))

        combined_pct = base_pct * corr_mult
        position_dollars = total_value * combined_pct

        pos = view.portfolio.positions.get(ticker)
        current_exposure = abs((pos.long - pos.short) * price) if pos else 0.0
        remaining = position_dollars - current_exposure
        max_position = min(remaining, view.portfolio.cash)
        max_shares = int(max_position // price) if price > 0 else 0

        limits.per_ticker[ticker] = TickerLimit(
            ticker=ticker,
            current_price=price,
            max_position_dollars=float(max(0.0, max_position)),
            max_shares=max(0, max_shares),
            annualized_volatility=float(annualized_vol),
            correlation_multiplier=float(corr_mult),
            remaining_position_limit=float(max(0.0, remaining)),
        )
        progress.update_status(
            AGENT_ID, None,
            f"{ticker}: adj. limit {combined_pct:.1%}, available ${max_position:.0f}",
        )

    progress.update_status(AGENT_ID, None, "Done")
    return limits
