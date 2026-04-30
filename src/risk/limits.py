"""Pure risk limit calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    """All risk limits in one place."""

    max_position_pct: float = 0.15
    min_position_pct: float = 0.02
    max_sector_pct: float = 0.40
    max_gross_exposure_pct: float = 1.5
    max_correlation_exposure: float = 0.50
    volatility_lookback_days: int = 60
    correlation_min_data_points: int = 5
    drawdown_reduction_threshold: float = 0.05
    drawdown_max_reduction: float = 0.50


def calculate_volatility_metrics(prices_df: pd.DataFrame, lookback_days: int = 60) -> dict:
    """Calculate volatility metrics from price data."""
    if len(prices_df) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * np.sqrt(252),
            "volatility_percentile": 100,
            "data_points": len(prices_df),
        }

    daily_returns = prices_df["close"].pct_change().dropna()
    if len(daily_returns) < 2:
        return {
            "daily_volatility": 0.05,
            "annualized_volatility": 0.05 * np.sqrt(252),
            "volatility_percentile": 100,
            "data_points": len(daily_returns),
        }

    recent_returns = daily_returns.tail(min(lookback_days, len(daily_returns)))
    daily_vol = recent_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)

    current_vol_percentile = 50.0
    if len(daily_returns) >= 30:
        rolling_vol = daily_returns.rolling(window=30).std().dropna()
        if len(rolling_vol) > 0:
            current_vol_percentile = float((rolling_vol <= daily_vol).mean() * 100)

    return {
        "daily_volatility": float(daily_vol) if not np.isnan(daily_vol) else 0.025,
        "annualized_volatility": float(annualized_vol) if not np.isnan(annualized_vol) else 0.25,
        "volatility_percentile": float(current_vol_percentile) if not np.isnan(current_vol_percentile) else 50.0,
        "data_points": len(recent_returns),
    }


def volatility_adjusted_limit(annualized_volatility: float, config: RiskConfig) -> float:
    """Position limit as % of portfolio based on volatility."""
    base = config.max_position_pct

    if annualized_volatility < 0.15:
        multiplier = 1.0
    elif annualized_volatility < 0.30:
        multiplier = 1.0 - (annualized_volatility - 0.15) * 2.0
    elif annualized_volatility < 0.50:
        multiplier = 0.70 - (annualized_volatility - 0.30) * 1.5
    else:
        multiplier = 0.40

    multiplier = max(0.20, min(1.0, multiplier))
    return base * multiplier


def correlation_multiplier(avg_correlation: float) -> float:
    """Reduce limit when highly correlated with active positions."""
    if avg_correlation >= 0.80:
        return 0.60
    if avg_correlation >= 0.60:
        return 0.75
    if avg_correlation >= 0.40:
        return 0.90
    if avg_correlation >= 0.20:
        return 1.0
    return 1.05


def compute_position_limit(
    *,
    portfolio_value: float,
    current_position_value: float,
    available_cash: float,
    gross_exposure_pct: float,
    annualized_volatility: float,
    avg_correlation: float | None,
    drawdown_multiplier: float,
    config: RiskConfig,
) -> dict:
    """Compute the remaining dollar headroom for a ticker."""
    vol_limit_pct = volatility_adjusted_limit(annualized_volatility, config)
    corr_multiplier = correlation_multiplier(avg_correlation) if avg_correlation is not None else 1.0

    gross_headroom_pct = max(0, config.max_gross_exposure_pct - gross_exposure_pct)
    gross_limit = portfolio_value * gross_headroom_pct

    combined_limit_pct = vol_limit_pct * corr_multiplier * drawdown_multiplier
    combined_limit_pct = min(combined_limit_pct, config.max_position_pct)
    position_limit = portfolio_value * combined_limit_pct
    position_limit = min(position_limit, gross_limit)

    remaining = position_limit - current_position_value
    max_position_size = min(remaining, available_cash)
    max_position_size = max(0, max_position_size)

    return {
        "remaining_position_limit": float(max_position_size),
        "vol_limit_pct": float(vol_limit_pct),
        "correlation_multiplier": float(corr_multiplier),
        "combined_limit_pct": float(combined_limit_pct),
        "position_limit": float(position_limit),
        "remaining": float(remaining),
    }
