"""Trade execution wrapper for simulated portfolios.

The executor maps normalized actions such as ``buy`` and ``short`` onto the
portfolio accounting methods while applying the configured slippage and
commission model.
"""

from __future__ import annotations

from .costs import CostModel
from .portfolio import Portfolio
from .types import ActionLiteral, Action


class TradeExecutor:
    """Executes trades against a Portfolio with Backtester-identical semantics."""

    def __init__(self, cost_model: CostModel | None = None) -> None:
        self.cost_model = cost_model or CostModel()

    def execute_trade(
        self,
        ticker: str,
        action: ActionLiteral,
        quantity: float,
        current_price: float,
        portfolio: Portfolio,
    ) -> int:
        if quantity is None or quantity <= 0:
            return 0

        # Coerce to enum if strings provided
        try:
            action_enum = Action(action) if not isinstance(action, Action) else action
        except Exception:
            action_enum = Action.HOLD

        executed_price = self.cost_model.execution_price(action_enum.value, float(current_price))
        fee = self.cost_model.commission(int(quantity))

        if action_enum == Action.BUY:
            return portfolio.apply_long_buy(ticker, int(quantity), executed_price, fee=fee)
        if action_enum == Action.SELL:
            return portfolio.apply_long_sell(ticker, int(quantity), executed_price, fee=fee)
        if action_enum == Action.SHORT:
            return portfolio.apply_short_open(ticker, int(quantity), executed_price, fee=fee)
        if action_enum == Action.COVER:
            return portfolio.apply_short_cover(ticker, int(quantity), executed_price, fee=fee)

        # hold or unknown action
        return 0
