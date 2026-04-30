"""Small technical helpers retained by the rebuilt core."""

from __future__ import annotations

import pandas as pd


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def calculate_fibonacci_signals(prices_df: pd.DataFrame) -> dict:
    """Return a Fibonacci retracement signal from price history."""
    close = prices_df["close"]

    lookback = min(len(close), 60)
    recent = close.iloc[-lookback:]
    swing_high = recent.max()
    swing_low = recent.min()
    swing_range = swing_high - swing_low

    if swing_range < 1e-8:
        return {
            "signal": "neutral",
            "confidence": 0.5,
            "metrics": {
                "swing_high": _safe_float(swing_high),
                "swing_low": _safe_float(swing_low),
                "nearest_fib_level": "N/A",
                "price_vs_range": 0.5,
            },
        }

    current_price = close.iloc[-1]
    high_idx = recent.idxmax()
    low_idx = recent.idxmin()
    uptrend = low_idx < high_idx

    fib_levels = {
        "0.0": swing_high if uptrend else swing_low,
        "23.6": swing_high - 0.236 * swing_range if uptrend else swing_low + 0.236 * swing_range,
        "38.2": swing_high - 0.382 * swing_range if uptrend else swing_low + 0.382 * swing_range,
        "50.0": swing_high - 0.500 * swing_range if uptrend else swing_low + 0.500 * swing_range,
        "61.8": swing_high - 0.618 * swing_range if uptrend else swing_low + 0.618 * swing_range,
        "78.6": swing_high - 0.786 * swing_range if uptrend else swing_low + 0.786 * swing_range,
        "100.0": swing_low if uptrend else swing_high,
    }

    if uptrend:
        retracement_pct = (swing_high - current_price) / swing_range
    else:
        retracement_pct = (current_price - swing_low) / swing_range

    fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    fib_names = ["0.0", "23.6", "38.2", "50.0", "61.8", "78.6", "100.0"]
    distances = [abs(retracement_pct - ratio) for ratio in fib_ratios]
    nearest_idx = distances.index(min(distances))
    nearest_fib = fib_names[nearest_idx]
    proximity = min(distances)
    at_fib_level = proximity < 0.02
    recent_direction = close.pct_change().iloc[-5:].mean()

    if uptrend:
        if 0.35 <= retracement_pct <= 0.65 and at_fib_level and recent_direction > 0:
            signal = "bullish"
            confidence = 0.7 + (0.3 * (1 - proximity / 0.02))
        elif retracement_pct > 0.786:
            signal = "bearish"
            confidence = 0.6
        elif retracement_pct < 0.236 and recent_direction > 0:
            signal = "bullish"
            confidence = 0.55
        else:
            signal = "neutral"
            confidence = 0.5
    else:
        if 0.35 <= retracement_pct <= 0.65 and at_fib_level and recent_direction < 0:
            signal = "bearish"
            confidence = 0.7 + (0.3 * (1 - proximity / 0.02))
        elif retracement_pct > 0.786:
            signal = "bullish"
            confidence = 0.6
        elif retracement_pct < 0.236 and recent_direction < 0:
            signal = "bearish"
            confidence = 0.55
        else:
            signal = "neutral"
            confidence = 0.5

    return {
        "signal": signal,
        "confidence": min(confidence, 1.0),
        "metrics": {
            "swing_high": _safe_float(swing_high),
            "swing_low": _safe_float(swing_low),
            "retracement_pct": _safe_float(retracement_pct),
            "nearest_fib_level": nearest_fib,
            "at_fib_level": at_fib_level,
            "trend_direction": "up" if uptrend else "down",
            "fib_levels": {name: _safe_float(value) for name, value in fib_levels.items()},
        },
    }
