"""Abstract broker interface for trade execution.

All broker integrations (Alpaca, IBKR, etc.) implement this interface.
The bridge module converts portfolio manager signals into broker orders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class Order:
    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    order_type: str = "market"
    status: str = "pending"
    filled_price: float | None = None
    filled_at: str | None = None
    order_id: str | None = None
    reasoning: str = ""


@dataclass
class Position:
    ticker: str
    quantity: int
    side: Literal["long", "short"]
    avg_cost: float
    market_value: float
    unrealized_pnl: float


@dataclass
class AccountSummary:
    equity: float
    cash: float
    buying_power: float
    positions: list[Position] = field(default_factory=list)


class BrokerBase(ABC):
    """Abstract base class for broker integrations."""

    @abstractmethod
    def place_order(self, ticker: str, side: str, quantity: int, order_type: str = "market") -> Order:
        """Place a trade order. Returns the Order with status updated."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all current open positions."""
        ...

    @abstractmethod
    def get_account(self) -> AccountSummary:
        """Get account summary (equity, cash, buying power)."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if cancelled successfully."""
        ...


def signal_to_orders(decisions: dict, broker: BrokerBase) -> list[Order]:
    """Convert portfolio manager decisions to broker orders.

    Args:
        decisions: Dict of {ticker: {action, quantity, confidence, reasoning}}
        broker: A BrokerBase implementation

    Returns:
        List of Order objects with execution results
    """
    orders = []
    for ticker, decision in decisions.items():
        action = decision.get("action", "hold")
        quantity = decision.get("quantity", 0)
        reasoning = decision.get("reasoning", "")

        if action == "hold" or quantity == 0:
            continue

        # Map agent actions to broker sides
        side_map = {
            "buy": "buy",
            "cover": "buy",    # Covering a short = buying
            "sell": "sell",
            "short": "sell",   # Opening a short = selling
        }

        side = side_map.get(action)
        if not side:
            continue

        order = broker.place_order(
            ticker=ticker,
            side=side,
            quantity=quantity,
        )
        order.reasoning = reasoning
        orders.append(order)

    return orders
