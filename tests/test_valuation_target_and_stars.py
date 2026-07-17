"""Valuation display and STARS rating behavior."""

from src.aggregator import _stars, compute_consensus
from src.schemas.signals import Signal


def test_stars_mapping_aligns_with_direction_cut():
    assert _stars(0.40)[0] == 5 and _stars(0.41) == (5, "Strong Buy")
    assert _stars(0.10) == (4, "Buy")
    assert _stars(0.0) == (3, "Hold")
    assert _stars(-0.10) == (2, "Sell")        # boundary is exclusive on the hold side
    assert _stars(-0.40) == (1, "Strong Sell")


def test_consensus_carries_stars():
    signals = [
        Signal(agent_id="a", ticker="NVDA", direction="bullish", confidence=0.9),
        Signal(agent_id="b", ticker="NVDA", direction="bullish", confidence=0.9),
    ]
    c = compute_consensus(signals, ["NVDA"])["NVDA"]
    assert c.stars == 5 and c.stars_label == "Strong Buy"


def test_empty_consensus_defaults_to_hold():
    c = compute_consensus([], ["NVDA"])["NVDA"]
    assert c.stars == 3 and c.stars_label == "Hold"


def test_context_omits_retired_12mo_target_language():
    from src.chat.signal_card import signal_cards_context_text
    from src.schemas.signals import Consensus, Decisions, Recommendation, Signal, TickerDecision

    rec = Recommendation(
        run_id="r1", as_of="2026-07-17", tickers=["META"],
        signals=[Signal(agent_id="valuation", ticker="META", direction="bearish",
                        confidence=0.95)],
        consensus={"META": Consensus(ticker="META", direction="bullish", confidence=0.64,
                                     weighted_score=0.3, stars=4, stars_label="Buy", bull_count=5)},
        decisions=Decisions(per_ticker={"META": TickerDecision(ticker="META", action="buy",
                                                               quantity=10, confidence=0.64)}),
    )
    ctx = signal_cards_context_text(rec)
    assert "12-month target" not in ctx
    assert "12mo target" not in ctx
