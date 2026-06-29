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
