"""Tests for Fibonacci retracement signal calculation."""

import pytest
import pandas as pd
import numpy as np

from src.alpha.features.technical import calculate_fibonacci_signals


def _make_prices_df(closes: list[float], high_pad: float = 1.0, low_pad: float = 1.0) -> pd.DataFrame:
    """Build a realistic OHLCV DataFrame from a list of close prices.

    high and low are padded around close to simulate intraday range.
    Volume is constant (not used by Fibonacci logic).
    """
    df = pd.DataFrame({
        "close": closes,
        "open": closes,
        "high": [c + high_pad for c in closes],
        "low": [c - low_pad for c in closes],
        "volume": [1_000_000] * len(closes),
    })
    return df


def _uptrend_then_retrace(retrace_pct: float, n_bars: int = 60) -> pd.DataFrame:
    """Create a price series that trends up then retraces by retrace_pct.

    First half: rises from 100 to 200 (swing low=100, swing high=200).
    Second half: retraces down by retrace_pct of the range (100 pts).
    Last 5 bars have slight upward drift (bounce confirmation).
    """
    half = n_bars // 2
    # Uptrend: 100 → 200
    up = np.linspace(100, 200, half)
    # Retrace target
    retrace_target = 200 - retrace_pct * 100  # range is 100
    # Retrace and then bounce slightly in last 5 bars
    retrace_bars = n_bars - half - 5
    down = np.linspace(200, retrace_target, max(retrace_bars, 1))
    # Small bounce in last 5 bars (confirms uptrend)
    bounce = np.linspace(retrace_target, retrace_target + 2, 5)
    closes = list(up) + list(down) + list(bounce)
    return _make_prices_df(closes[:n_bars])


def _downtrend_then_retrace(retrace_pct: float, n_bars: int = 60) -> pd.DataFrame:
    """Create a price series that trends down then retraces up.

    First half: drops from 200 to 100 (swing high=200, swing low=100).
    Second half: retraces up by retrace_pct of the range.
    Last 5 bars have slight downward drift (bearish confirmation).
    """
    half = n_bars // 2
    down = np.linspace(200, 100, half)
    retrace_target = 100 + retrace_pct * 100
    retrace_bars = n_bars - half - 5
    up = np.linspace(100, retrace_target, max(retrace_bars, 1))
    # Small drop in last 5 bars (confirms downtrend)
    drop = np.linspace(retrace_target, retrace_target - 2, 5)
    closes = list(down) + list(up) + list(drop)
    return _make_prices_df(closes[:n_bars])


class TestUptrendRetracement:
    def test_38pct_retracement_bullish(self):
        """Price retracing ~38% in uptrend near fib level → bullish."""
        df = _uptrend_then_retrace(retrace_pct=0.382)
        result = calculate_fibonacci_signals(df)
        assert result["metrics"]["trend_direction"] == "up"
        # Retracement should be close to 38.2%
        assert 0.30 <= result["metrics"]["retracement_pct"] <= 0.50
        # Should be bullish (at fib level with bounce) or neutral
        assert result["signal"] in ("bullish", "neutral")

    def test_50pct_retracement_bullish(self):
        """50% retracement in uptrend is a classic buy zone."""
        df = _uptrend_then_retrace(retrace_pct=0.50)
        result = calculate_fibonacci_signals(df)
        assert result["metrics"]["trend_direction"] == "up"
        assert 0.40 <= result["metrics"]["retracement_pct"] <= 0.60

    def test_shallow_retracement_near_high(self):
        """< 23.6% retracement, price near swing high, still strong."""
        df = _uptrend_then_retrace(retrace_pct=0.10)
        result = calculate_fibonacci_signals(df)
        assert result["metrics"]["trend_direction"] == "up"
        assert result["metrics"]["retracement_pct"] < 0.236
        # Should be bullish or neutral (near top of range)
        assert result["signal"] in ("bullish", "neutral")


class TestDowntrendRetracement:
    def test_38pct_retrace_in_downtrend(self):
        """Retracement in downtrend near 38.2% → bearish or neutral."""
        df = _downtrend_then_retrace(retrace_pct=0.382)
        result = calculate_fibonacci_signals(df)
        assert result["metrics"]["trend_direction"] == "down"
        assert 0.30 <= result["metrics"]["retracement_pct"] <= 0.50
        assert result["signal"] in ("bearish", "neutral")

    def test_shallow_retrace_downtrend(self):
        """Shallow retrace in downtrend, price near swing low."""
        df = _downtrend_then_retrace(retrace_pct=0.10)
        result = calculate_fibonacci_signals(df)
        assert result["metrics"]["trend_direction"] == "down"
        assert result["metrics"]["retracement_pct"] < 0.25
        assert result["signal"] in ("bearish", "neutral")


class TestDeepRetracement:
    def test_deep_retracement_uptrend_bearish(self):
        """> 78.6% retracement in uptrend → trend may be broken → bearish."""
        df = _uptrend_then_retrace(retrace_pct=0.85)
        result = calculate_fibonacci_signals(df)
        assert result["metrics"]["trend_direction"] == "up"
        assert result["metrics"]["retracement_pct"] > 0.75
        assert result["signal"] == "bearish"
        assert result["confidence"] >= 0.5

    def test_deep_retracement_downtrend_bullish(self):
        """> 78.6% retracement in downtrend → trend broken → bullish."""
        df = _downtrend_then_retrace(retrace_pct=0.85)
        result = calculate_fibonacci_signals(df)
        assert result["metrics"]["trend_direction"] == "down"
        assert result["metrics"]["retracement_pct"] > 0.75
        assert result["signal"] == "bullish"
        assert result["confidence"] >= 0.5


class TestNoRange:
    def test_flat_price_neutral(self):
        """Flat price (no range) → neutral with default confidence."""
        closes = [100.0] * 60
        # Use very small padding so high/low are almost the same
        df = _make_prices_df(closes, high_pad=0.0, low_pad=0.0)
        result = calculate_fibonacci_signals(df)
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.5
        assert result["metrics"]["nearest_fib_level"] == "N/A"


class TestFibLevels:
    def test_fib_levels_present(self):
        """Metrics should contain all Fibonacci levels."""
        df = _uptrend_then_retrace(retrace_pct=0.50)
        result = calculate_fibonacci_signals(df)
        fib_levels = result["metrics"]["fib_levels"]
        for key in ["0.0", "23.6", "38.2", "50.0", "61.8", "78.6", "100.0"]:
            assert key in fib_levels

    def test_confidence_capped_at_1(self):
        """Confidence should never exceed 1.0."""
        df = _uptrend_then_retrace(retrace_pct=0.382)
        result = calculate_fibonacci_signals(df)
        assert result["confidence"] <= 1.0

    def test_returns_valid_signal_strings(self):
        """Signal must be one of the valid values."""
        for pct in [0.10, 0.382, 0.50, 0.618, 0.85]:
            df = _uptrend_then_retrace(retrace_pct=pct)
            result = calculate_fibonacci_signals(df)
            assert result["signal"] in ("bullish", "bearish", "neutral")
