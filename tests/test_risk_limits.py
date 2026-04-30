"""Unit tests for portfolio risk limit calculations.

This module checks the small risk helper functions that shrink or cap position
size based on volatility, correlation, available cash, and existing exposure.
"""

from src.risk.limits import RiskConfig, compute_position_limit, correlation_multiplier, volatility_adjusted_limit


def test_volatility_adjusted_limit_declines_for_high_vol():
    config = RiskConfig()
    low = volatility_adjusted_limit(0.10, config)
    high = volatility_adjusted_limit(0.60, config)

    assert low > high


def test_correlation_multiplier_penalizes_high_correlation():
    assert correlation_multiplier(0.85) < correlation_multiplier(0.10)


def test_compute_position_limit_respects_cash_and_position():
    result = compute_position_limit(
        portfolio_value=100000,
        current_position_value=5000,
        available_cash=7000,
        gross_exposure_pct=0.20,
        annualized_volatility=0.20,
        avg_correlation=0.30,
        drawdown_multiplier=1.0,
        config=RiskConfig(),
    )

    assert result["remaining_position_limit"] <= 7000
    assert result["position_limit"] >= result["remaining_position_limit"]
