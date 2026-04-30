"""Tests for deterministic weighted voting system."""

import pytest

from src.portfolio.construction import VotingConfig, VoteResult, vote_on_ticker


@pytest.fixture
def default_config():
    return VotingConfig()


@pytest.fixture
def default_position():
    return {"long": 0, "short": 0}


def _make_signal(signal: str, confidence: float, status: str = "success") -> dict:
    return {"signal": signal, "confidence": confidence, "status": status}


class TestBullishConsensus:
    def test_all_agents_bullish_produces_buy(self, default_config, default_position):
        signals = {
            "agent_a": _make_signal("bullish", 80),
            "agent_b": _make_signal("bullish", 90),
            "agent_c": _make_signal("bullish", 70),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "buy"
        assert result.weighted_score > 0
        assert result.bull_count == 3
        assert result.bear_count == 0
        assert result.quantity > 0

    def test_bullish_consensus_covers_short(self, default_config):
        signals = {
            "agent_a": _make_signal("bullish", 80),
            "agent_b": _make_signal("bullish", 90),
        }
        position = {"long": 0, "short": 50}
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=position, config=default_config)
        assert result.action == "cover"
        assert result.quantity == 50


class TestBearishConsensus:
    def test_all_agents_bearish_produces_short(self, default_config, default_position):
        signals = {
            "agent_a": _make_signal("bearish", 80),
            "agent_b": _make_signal("bearish", 90),
            "agent_c": _make_signal("bearish", 70),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "short"
        assert result.weighted_score < 0
        assert result.bear_count == 3

    def test_bearish_consensus_sells_long(self, default_config):
        signals = {
            "agent_a": _make_signal("bearish", 80),
            "agent_b": _make_signal("bearish", 90),
        }
        position = {"long": 30, "short": 0}
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=position, config=default_config)
        assert result.action == "sell"
        assert result.quantity == 30


class TestMixedSignals:
    def test_mixed_signals_near_threshold(self, default_config, default_position):
        """Two bullish, one bearish — score should still cross weak threshold."""
        signals = {
            "agent_a": _make_signal("bullish", 80),
            "agent_b": _make_signal("bullish", 70),
            "agent_c": _make_signal("bearish", 60),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        # 2/3 bullish = 66% consensus, score positive
        assert result.action == "buy"
        assert result.bull_count == 2
        assert result.bear_count == 1

    def test_evenly_split_signals_hold(self, default_config, default_position):
        """Equal bullish and bearish cancel out → hold."""
        signals = {
            "agent_a": _make_signal("bullish", 80),
            "agent_b": _make_signal("bearish", 80),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "hold"
        assert result.quantity == 0

    def test_mostly_neutral_hold(self, default_config, default_position):
        """Mostly neutral signals → hold even if one is bullish."""
        signals = {
            "agent_a": _make_signal("neutral", 50),
            "agent_b": _make_signal("neutral", 50),
            "agent_c": _make_signal("neutral", 50),
            "agent_d": _make_signal("bullish", 60),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        # Only 1/4 = 25% bullish, below 30% consensus
        assert result.action == "hold"


class TestFailedAgentsExcluded:
    def test_failed_agents_not_counted(self, default_config, default_position):
        signals = {
            "agent_a": _make_signal("bullish", 80),
            "agent_b": _make_signal("bullish", 90),
            "agent_c": _make_signal("bearish", 70, status="failed"),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "buy"
        assert result.failed_count == 1
        assert result.total_voters == 2  # Only 2 usable
        assert result.bull_count == 2
        assert result.bear_count == 0

    def test_all_failed_produces_hold(self, default_config, default_position):
        signals = {
            "agent_a": _make_signal("bullish", 80, status="failed"),
            "agent_b": _make_signal("bearish", 90, status="failed"),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "hold"
        assert result.quantity == 0
        assert result.failed_count == 2
        assert result.total_voters == 0


class TestLowConfidenceFiltered:
    def test_below_threshold_ignored(self, default_config, default_position):
        """Signals below min_confidence_to_count (30) are dropped."""
        signals = {
            "agent_a": _make_signal("bullish", 80),
            "agent_b": _make_signal("bearish", 10),  # Below 30 → filtered
            "agent_c": _make_signal("bearish", 20),  # Below 30 → filtered
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "buy"
        assert result.total_voters == 1  # Only agent_a counted

    def test_all_low_confidence_hold(self, default_config, default_position):
        signals = {
            "agent_a": _make_signal("bullish", 5),
            "agent_b": _make_signal("bearish", 10),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "hold"
        assert result.total_voters == 0


class TestEmptySignals:
    def test_no_signals_hold(self, default_config, default_position):
        result = vote_on_ticker("AAPL", {}, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.action == "hold"
        assert result.quantity == 0
        assert result.total_voters == 0


class TestPositionSizing:
    def test_stronger_signal_larger_position(self, default_position):
        """Stronger consensus → more shares (proportional sizing)."""
        # Strong signal
        strong_signals = {
            "a": _make_signal("bullish", 95),
            "b": _make_signal("bullish", 95),
            "c": _make_signal("bullish", 95),
        }
        strong = vote_on_ticker("AAPL", strong_signals, max_shares=100,
                                current_price=150.0, position=default_position)

        # Weaker signal
        weak_signals = {
            "a": _make_signal("bullish", 40),
            "b": _make_signal("bullish", 40),
            "c": _make_signal("neutral", 40),
        }
        weak = vote_on_ticker("AAPL", weak_signals, max_shares=100,
                              current_price=150.0, position=default_position)

        assert strong.action == "buy"
        assert weak.action in ("buy", "hold")
        if weak.action == "buy":
            assert strong.quantity > weak.quantity

    def test_quantity_never_exceeds_max_shares(self, default_config, default_position):
        signals = {
            "a": _make_signal("bullish", 100),
            "b": _make_signal("bullish", 100),
        }
        result = vote_on_ticker("AAPL", signals, max_shares=50, current_price=150.0,
                                position=default_position, config=default_config)
        assert result.quantity <= 50


class TestVoteResultSerialization:
    def test_to_dict(self, default_config, default_position):
        signals = {"a": _make_signal("bullish", 80)}
        result = vote_on_ticker("AAPL", signals, max_shares=100, current_price=150.0,
                                position=default_position, config=default_config)
        d = result.to_dict()
        assert "action" in d
        assert "quantity" in d
        assert "confidence" in d
        assert "reasoning" in d
        assert "vote_details" in d
        assert "weighted_score" in d["vote_details"]
