"""Tests for circuit breaker safety systems."""

import time
import pytest
from unittest.mock import patch

from src.execution.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    PortfolioSnapshot,
    KILL_SWITCH_PATH,
    CIRCUIT_TRIPPED_PATH,
)


@pytest.fixture
def config():
    return CircuitBreakerConfig(
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_trade_notional_pct=0.05,
        max_position_pct=0.20,
        max_orders_per_minute=10,
        min_confidence_to_trade=60.0,
    )


@pytest.fixture
def cb(config):
    # Ensure no stale persisted trip state affects tests
    CIRCUIT_TRIPPED_PATH.unlink(missing_ok=True)
    breaker = CircuitBreaker(config=config)
    yield breaker
    CIRCUIT_TRIPPED_PATH.unlink(missing_ok=True)


@pytest.fixture
def healthy_portfolio():
    """Portfolio in good shape: NAV=100k, no loss, no drawdown."""
    return PortfolioSnapshot(
        nav=100_000,
        cash=50_000,
        positions={"AAPL": {"long": 100, "value": 15_000}},
        day_start_nav=100_000,
        peak_nav=100_000,
    )


class TestKillSwitch:
    def test_kill_switch_blocks_all(self, cb, healthy_portfolio):
        with patch.object(CircuitBreaker, "is_killed", return_value=True):
            ok, reason = cb.check_order("AAPL", "buy", 10, 150.0, healthy_portfolio)
            assert ok is False
            assert "KILL SWITCH" in reason

    def test_no_kill_switch_allows(self, cb, healthy_portfolio):
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            ok, reason = cb.check_order("MSFT", "buy", 10, 100.0, healthy_portfolio)
            assert ok is True
            assert reason == ""


class TestDailyLossLimit:
    def test_daily_loss_blocks(self, cb):
        """When daily loss exceeds 2%, orders are blocked."""
        portfolio = PortfolioSnapshot(
            nav=97_000,  # Lost 3% from day start
            cash=50_000,
            positions={},
            day_start_nav=100_000,
            peak_nav=100_000,
        )
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            ok, reason = cb.check_order("AAPL", "buy", 1, 100.0, portfolio)
            assert ok is False
            assert "Daily loss" in reason

    def test_within_daily_loss_allowed(self, cb):
        """Small daily loss (1%) should not trigger."""
        portfolio = PortfolioSnapshot(
            nav=99_000,
            cash=50_000,
            positions={},
            day_start_nav=100_000,
            peak_nav=100_000,
        )
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            ok, _ = cb.check_order("AAPL", "buy", 1, 100.0, portfolio)
            assert ok is True


class TestMaxPositionSize:
    def test_position_exceeds_limit(self, cb):
        """Single position cannot exceed 20% of NAV."""
        portfolio = PortfolioSnapshot(
            nav=100_000,
            cash=80_000,
            positions={"AAPL": {"long": 100, "value": 18_000}},
            day_start_nav=100_000,
            peak_nav=100_000,
        )
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            # Trying to add $3k to existing $18k AAPL = $21k > 20% of $100k
            # Trade notional $3k < 5% of NAV ($5k), so notional check passes
            ok, reason = cb.check_order("AAPL", "buy", 30, 100.0, portfolio)
            assert ok is False
            assert "Position" in reason

    def test_position_within_limit(self, cb, healthy_portfolio):
        """Position within limit should pass."""
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            # $1000 + existing $15k = $16k < $20k limit
            ok, _ = cb.check_order("AAPL", "buy", 10, 100.0, healthy_portfolio)
            assert ok is True


class TestRateLimiter:
    def test_rate_limit_exceeded(self, config):
        config.max_orders_per_minute = 3
        cb = CircuitBreaker(config=config)
        portfolio = PortfolioSnapshot(
            nav=100_000, cash=90_000, positions={},
            day_start_nav=100_000, peak_nav=100_000,
        )
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            # Place 3 orders (should succeed)
            for _ in range(3):
                ok, _ = cb.check_order("AAPL", "buy", 1, 100.0, portfolio)
                assert ok is True

            # 4th order should be rate limited
            ok, reason = cb.check_order("AAPL", "buy", 1, 100.0, portfolio)
            assert ok is False
            assert "Rate limit" in reason


class TestNormalOrderPasses:
    def test_valid_order_passes_all_checks(self, cb, healthy_portfolio):
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            ok, reason = cb.check_order(
                "MSFT", "buy", 10, 100.0, healthy_portfolio, signal_confidence=80.0,
            )
            assert ok is True
            assert reason == ""


class TestDrawdownBreaker:
    def test_blocks_when_drawdown_too_deep(self, cb):
        """Drawdown > 10% from peak trips the breaker."""
        portfolio = PortfolioSnapshot(
            nav=88_000,  # -12% from peak
            cash=50_000,
            positions={},
            day_start_nav=88_000,  # Day start same as current (no daily loss)
            peak_nav=100_000,
        )
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            ok, reason = cb.check_order("AAPL", "buy", 1, 100.0, portfolio)
            assert ok is False
            assert "Drawdown" in reason

    def test_drawdown_within_limit(self, cb):
        """Drawdown of 5% should be fine (limit is 10%)."""
        portfolio = PortfolioSnapshot(
            nav=95_000,
            cash=50_000,
            positions={},
            day_start_nav=95_000,
            peak_nav=100_000,
        )
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            ok, _ = cb.check_order("AAPL", "buy", 1, 100.0, portfolio)
            assert ok is True


class TestTradeNotionalCap:
    def test_trade_too_large(self, cb, healthy_portfolio):
        """Single trade > 5% of NAV blocked."""
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            # $6000 > 5% of $100k = $5000
            ok, reason = cb.check_order("MSFT", "buy", 30, 200.0, healthy_portfolio)
            assert ok is False
            assert "notional" in reason.lower()


class TestMinConfidence:
    def test_low_confidence_blocked(self, cb, healthy_portfolio):
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            ok, reason = cb.check_order(
                "AAPL", "buy", 1, 100.0, healthy_portfolio, signal_confidence=40.0,
            )
            assert ok is False
            assert "confidence" in reason.lower()


class TestCircuitBreakerTrip:
    def test_tripped_breaker_blocks_subsequent(self, cb):
        """Once tripped, all subsequent orders are blocked."""
        portfolio = PortfolioSnapshot(
            nav=97_000, cash=50_000, positions={},
            day_start_nav=100_000, peak_nav=100_000,
        )
        with patch.object(CircuitBreaker, "is_killed", return_value=False):
            # First call trips the breaker (daily loss)
            ok1, _ = cb.check_order("AAPL", "buy", 1, 100.0, portfolio)
            assert ok1 is False
            assert cb.is_tripped

            # Now even a valid order on a healthy portfolio is blocked
            healthy = PortfolioSnapshot(
                nav=100_000, cash=90_000, positions={},
                day_start_nav=100_000, peak_nav=100_000,
            )
            ok2, reason = cb.check_order("AAPL", "buy", 1, 100.0, healthy)
            assert ok2 is False
            assert "Circuit breaker tripped" in reason

    def test_reset_clears_trip(self, cb):
        """Reset should clear tripped state."""
        cb._trip("test", "test reason")
        assert cb.is_tripped
        cb.reset()
        assert not cb.is_tripped
