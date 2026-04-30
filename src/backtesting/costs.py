"""Trading cost model used by the backtesting stack.

This file keeps slippage and commission rules in one place so trade simulation
uses the same assumptions everywhere in tests, backtests, and future strategy
evaluation work.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CostModel:
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    min_commission: float = 1.0

    def execution_price(self, action: str, reference_price: float) -> float:
        slip = reference_price * (self.slippage_bps / 10000.0)
        if action in ("buy", "cover"):
            return reference_price + slip
        if action in ("sell", "short"):
            return max(0.0, reference_price - slip)
        return reference_price

    def commission(self, quantity: int) -> float:
        if quantity <= 0:
            return 0.0
        return max(self.min_commission, quantity * self.commission_per_share)
