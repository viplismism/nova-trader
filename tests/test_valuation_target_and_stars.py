"""Step 8 (12-month price target) and Step 9 (STARS rating) — additive outputs."""

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


def test_12mo_target_is_fair_value_drifted_by_cost_of_equity():
    # fair value = price * (1 + gap); target = fair value * (1 + cost_of_equity)
    from src.schemas.signals import ValuationTarget

    vt = ValuationTarget(
        current_price=100.0,
        fair_value=118.0,          # +18% intrinsic gap
        target_price=118.0 * 1.105,  # drifted one year at 10.5% cost of equity
        upside=118.0 * 1.105 / 100.0 - 1,
        cost_of_equity=0.105,
    )
    assert round(vt.target_price, 2) == 130.39
    assert round(vt.upside, 4) == 0.3039


def test_context_flags_valuation_dissent_on_bullish_consensus():
    # A bearish valuation target under a bullish consensus must read as the
    # analyst's dissent, never as the desk's own target (VC-meeting lesson).
    from src.chat.signal_card import signal_cards_context_text
    from src.schemas.signals import (
        Consensus, Decisions, Recommendation, Signal, TickerDecision, ValuationTarget,
    )

    vt = ValuationTarget(current_price=633.0, fair_value=269.29, target_price=297.57,
                         upside=-0.53, cost_of_equity=0.105)
    rec = Recommendation(
        run_id="r1", as_of="2026-07-17", tickers=["META"],
        signals=[Signal(agent_id="valuation", ticker="META", direction="bearish",
                        confidence=0.95, valuation_target=vt)],
        consensus={"META": Consensus(ticker="META", direction="bullish", confidence=0.64,
                                     weighted_score=0.3, stars=4, stars_label="Buy", bull_count=5)},
        decisions=Decisions(per_ticker={"META": TickerDecision(ticker="META", action="buy",
                                                               quantity=10, confidence=0.64)}),
    )
    ctx = signal_cards_context_text(rec)
    assert "DISAGREES with the bullish consensus" in ctx
    assert "dissent" in ctx
    assert "valuation variant" in ctx
    # must NOT present the conservative model as the desk's own price target
    assert "desk's own price target" in ctx or "not the desk" in ctx
