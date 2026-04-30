"""Tests for SignalResult and signal health checking."""

import pytest

from src.core.signals import SignalResult, count_signal_health, signals_healthy_enough


class TestSignalSuccess:
    def test_success_fields(self):
        s = SignalResult.success(
            signal="bullish", confidence=85,
            reasoning={"score": 0.82}, agent_id="buffett",
            data_sources=["prices", "financials"],
        )
        assert s.signal == "bullish"
        assert s.confidence == 85
        assert s.status == "success"
        assert s.reasoning == {"score": 0.82}
        assert s.agent_id == "buffett"
        assert s.data_sources_used == ["prices", "financials"]
        assert s.error is None
        assert s.is_usable is True

    def test_success_to_dict(self):
        s = SignalResult.success(signal="bearish", confidence=60)
        d = s.to_dict()
        assert d["signal"] == "bearish"
        assert d["confidence"] == 60
        assert d["status"] == "success"
        assert "error" not in d  # No error field when None


class TestSignalDegraded:
    def test_degraded_reduces_confidence(self):
        s = SignalResult.degraded(
            signal="bullish", confidence=80,
            error="Missing insider data", agent_id="soros",
        )
        assert s.status == "degraded"
        assert s.confidence == 80 * 0.7  # 56.0
        assert s.error == "Missing insider data"
        assert s.is_usable is True

    def test_degraded_to_dict_includes_error(self):
        s = SignalResult.degraded(signal="neutral", confidence=50, error="Partial data")
        d = s.to_dict()
        assert d["status"] == "degraded"
        assert d["error"] == "Partial data"


class TestSignalFailed:
    def test_failed_fields(self):
        s = SignalResult.failed(error="API timeout", agent_id="technicals")
        assert s.signal is None
        assert s.confidence == 0
        assert s.status == "failed"
        assert s.error == "API timeout"
        assert s.is_usable is False

    def test_failed_to_dict_signal_defaults_neutral(self):
        s = SignalResult.failed(error="crash")
        d = s.to_dict()
        assert d["signal"] == "neutral"  # None → "neutral" in to_dict
        assert d["confidence"] == 0
        assert d["error"] == "crash"


class TestCountSignalHealth:
    def test_counts_correct(self):
        signals = {
            "agent_a": {
                "AAPL": {"signal": "bullish", "confidence": 80, "status": "success"},
                "MSFT": {"signal": "bearish", "confidence": 70, "status": "success"},
            },
            "agent_b": {
                "AAPL": {"signal": "neutral", "confidence": 0, "status": "failed"},
                "MSFT": {"signal": "bullish", "confidence": 50, "status": "degraded"},
            },
        }
        health = count_signal_health(signals)
        assert health["success"] == 2
        assert health["degraded"] == 1
        assert health["failed"] == 1
        assert health["total"] == 4

    def test_skips_risk_management(self):
        signals = {
            "risk_management_agent": {"AAPL": {"status": "success"}},
            "agent_a": {"AAPL": {"signal": "bullish", "status": "success"}},
        }
        health = count_signal_health(signals)
        assert health["total"] == 1  # Risk agent excluded

    def test_legacy_signals_assumed_success(self):
        """Signals without status field default to 'success'."""
        signals = {
            "agent_a": {"AAPL": {"signal": "bullish", "confidence": 80}},
        }
        health = count_signal_health(signals)
        assert health["success"] == 1
        assert health["failed"] == 0

    def test_empty_signals(self):
        health = count_signal_health({})
        assert health["total"] == 0


class TestSignalsHealthyEnough:
    def test_majority_success_healthy(self):
        signals = {
            "a": {"AAPL": {"status": "success"}},
            "b": {"AAPL": {"status": "success"}},
            "c": {"AAPL": {"status": "failed"}},
        }
        assert signals_healthy_enough(signals) is True  # 1/3 failed = 33%

    def test_majority_failed_unhealthy(self):
        signals = {
            "a": {"AAPL": {"status": "failed"}},
            "b": {"AAPL": {"status": "failed"}},
            "c": {"AAPL": {"status": "success"}},
        }
        assert signals_healthy_enough(signals) is False  # 2/3 failed = 66%

    def test_exactly_half_failed_healthy(self):
        """50% failure rate is at the boundary — should still be healthy (<=0.5)."""
        signals = {
            "a": {"AAPL": {"status": "success"}},
            "b": {"AAPL": {"status": "failed"}},
        }
        assert signals_healthy_enough(signals) is True

    def test_empty_signals_unhealthy(self):
        assert signals_healthy_enough({}) is False

    def test_custom_threshold(self):
        signals = {
            "a": {"AAPL": {"status": "failed"}},
            "b": {"AAPL": {"status": "success"}},
            "c": {"AAPL": {"status": "success"}},
            "d": {"AAPL": {"status": "success"}},
        }
        # 25% failed, stricter threshold of 20% should fail
        assert signals_healthy_enough(signals, max_failure_pct=0.20) is False
        # Default 50% threshold should pass
        assert signals_healthy_enough(signals) is True
