"""Alpaca paper trading integration.

Setup:
  1. pip install alpaca-py  (or uncomment in pyproject.toml)
  2. Create a free paper trading account at https://alpaca.markets
  3. Set these environment variables:
     ALPACA_API_KEY=your-paper-api-key
     ALPACA_SECRET_KEY=your-paper-secret-key
     ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""

import os
from src.execution.base import BrokerBase, Order, Position, AccountSummary


class AlpacaBroker(BrokerBase):
    """Alpaca paper trading broker.

    Wraps alpaca-py SDK for paper trading execution.
    """

    def __init__(self):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
        except ImportError:
            raise ImportError(
                "alpaca-py is not installed. Run: pip install alpaca-py\n"
                "Then set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env"
            )

        api_key = os.environ.get("ALPACA_API_KEY")
        secret_key = os.environ.get("ALPACA_SECRET_KEY")
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment. "
                "Get free paper trading keys at https://alpaca.markets"
            )

        self._client = TradingClient(api_key, secret_key, paper=True, url_override=base_url)
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._MarketOrderRequest = MarketOrderRequest

    def place_order(self, ticker: str, side: str, quantity: int, order_type: str = "market") -> Order:
        order_side = self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL

        request = self._MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=order_side,
            time_in_force=self._TimeInForce.DAY,
        )

        result = self._client.submit_order(request)
        return Order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=str(result.status),
            order_id=str(result.id),
        )

    def get_positions(self) -> list[Position]:
        positions = self._client.get_all_positions()
        return [
            Position(
                ticker=p.symbol,
                quantity=int(p.qty),
                side="long" if float(p.qty) > 0 else "short",
                avg_cost=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
            )
            for p in positions
        ]

    def get_account(self) -> AccountSummary:
        account = self._client.get_account()
        return AccountSummary(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            positions=self.get_positions(),
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False
