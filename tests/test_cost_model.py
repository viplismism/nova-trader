"""Unit tests for the backtesting cost model.

The assertions here keep the slippage and commission rules explicit so changes
to execution assumptions do not silently alter simulated trade behavior.
"""

from src.backtesting.costs import CostModel


def test_execution_price_applies_slippage_by_side():
    model = CostModel(slippage_bps=10.0, commission_per_share=0.0, min_commission=0.0)

    assert model.execution_price("buy", 100.0) > 100.0
    assert model.execution_price("sell", 100.0) < 100.0


def test_commission_has_minimum():
    model = CostModel(slippage_bps=0.0, commission_per_share=0.001, min_commission=1.0)

    assert model.commission(10) == 1.0
