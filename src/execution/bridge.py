"""Execution bridge — connects analysis decisions to broker execution.

Supports three modes:
  - dry_run: Log decisions, no broker interaction (default)
  - paper:   Execute via Alpaca paper trading
  - live:    Execute via Alpaca live trading (requires explicit opt-in)

Usage:
    bridge = ExecutionBridge.paper()  # or .dry_run()
    orders = bridge.execute(pad.decisions)
"""

from dataclasses import dataclass, field
from typing import Literal

from src.execution.base import BrokerBase, Order


@dataclass
class ExecutionResult:
    """Summary of an execution run."""
    mode: str
    orders: list[Order] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for o in self.orders if o.status not in ("failed", "rejected"))

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.orders if o.status in ("failed", "rejected"))


class ExecutionBridge:
    """Bridge between portfolio decisions and broker execution."""

    def __init__(self, mode: Literal["dry_run", "paper", "live"] = "dry_run", broker: BrokerBase | None = None):
        self.mode = mode
        self.broker = broker

        if mode in ("paper", "live") and broker is None:
            raise ValueError(f"Broker required for mode={mode}")

    @classmethod
    def dry_run(cls) -> "ExecutionBridge":
        """Create a dry-run bridge (no real orders)."""
        return cls(mode="dry_run")

    @classmethod
    def paper(cls) -> "ExecutionBridge":
        """Create a paper-trading bridge using Alpaca."""
        from src.execution.alpaca import AlpacaBroker
        return cls(mode="paper", broker=AlpacaBroker())

    def execute(self, decisions: dict) -> ExecutionResult:
        """Execute portfolio decisions.

        Args:
            decisions: Dict of {ticker: {action, quantity, confidence, reasoning}}

        Returns:
            ExecutionResult with order details.
        """
        result = ExecutionResult(mode=self.mode)

        for ticker, decision in decisions.items():
            action = decision.get("action", "hold")
            quantity = decision.get("quantity", 0)
            reasoning = decision.get("reasoning", "")

            if action == "hold" or quantity == 0:
                continue

            side_map = {
                "buy": "buy",
                "cover": "buy",
                "sell": "sell",
                "short": "sell",
            }
            side = side_map.get(action)
            if not side:
                continue

            if self.mode == "dry_run":
                order = Order(
                    ticker=ticker,
                    side=side,
                    quantity=quantity,
                    status="simulated",
                    reasoning=reasoning,
                )
                result.orders.append(order)
            else:
                try:
                    order = self.broker.place_order(
                        ticker=ticker,
                        side=side,
                        quantity=quantity,
                    )
                    order.reasoning = reasoning
                    result.orders.append(order)
                except Exception as e:
                    result.errors.append(f"{ticker}: {e}")
                    result.orders.append(Order(
                        ticker=ticker,
                        side=side,
                        quantity=quantity,
                        status="failed",
                        reasoning=str(e),
                    ))

        return result
