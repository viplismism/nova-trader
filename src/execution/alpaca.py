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
from src.utils.logger import get_logger

log = get_logger(__name__)

# Timeout for Alpaca API calls (seconds)
_API_TIMEOUT = 30


class AlpacaBroker(BrokerBase):
    """Alpaca paper trading broker.

    Wraps alpaca-py SDK for paper trading execution.
    All API calls are wrapped with error handling so network failures
    produce clear error messages rather than raw stack traces.
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

        if not api_key or not secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment. "
                "Get free paper trading keys at https://alpaca.markets"
            )

        # paper=True tells the SDK to use paper-api.alpaca.markets automatically
        self._client = TradingClient(api_key, secret_key, paper=True)
        self._OrderSide = OrderSide
        self._TimeInForce = TimeInForce
        self._MarketOrderRequest = MarketOrderRequest

    def place_order(self, ticker: str, side: str, quantity: int, order_type: str = "market") -> Order:
        if quantity <= 0:
            raise ValueError(f"Invalid quantity {quantity} for {ticker}: must be positive")
        if side not in ("buy", "sell"):
            raise ValueError(f"Invalid side '{side}': must be 'buy' or 'sell'")

        order_side = self._OrderSide.BUY if side == "buy" else self._OrderSide.SELL

        request = self._MarketOrderRequest(
            symbol=ticker,
            qty=quantity,
            side=order_side,
            time_in_force=self._TimeInForce.DAY,
        )

        try:
            result = self._client.submit_order(request)
        except Exception as e:
            log.error("Alpaca order submission failed for %s: %s", ticker, e, ticker=ticker)
            raise RuntimeError(f"Alpaca API error placing {side} {quantity} {ticker}: {e}") from e

        return Order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            order_type=order_type,
            status=str(result.status),
            order_id=str(result.id),
        )

    def get_positions(self) -> list[Position]:
        try:
            positions = self._client.get_all_positions()
        except Exception as e:
            log.error("Alpaca get_positions failed: %s", e)
            raise RuntimeError(f"Alpaca API error fetching positions: {e}") from e

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
        try:
            account = self._client.get_account()
        except Exception as e:
            log.error("Alpaca get_account failed: %s", e)
            raise RuntimeError(f"Alpaca API error fetching account: {e}") from e

        return AccountSummary(
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            positions=self.get_positions(),
        )

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        try:
            statuses = self._client.cancel_orders()
            return len(statuses) if statuses else 0
        except Exception as e:
            log.warning("Alpaca cancel_all_orders failed: %s", e)
            return 0

    def close_all_positions(self) -> list[dict]:
        """Cancel all orders, then close all open positions."""
        # Step 1: Cancel all pending orders first
        cancelled = self.cancel_all_orders()
        if cancelled:
            log.info("Cancelled %d pending orders before closing positions", cancelled)
            import time
            time.sleep(1)  # Give Alpaca a moment to process cancellations

        # Step 2: Close all positions
        try:
            responses = self._client.close_all_positions(cancel_orders=False)
            closed = []
            for resp in responses:
                order = resp.body if hasattr(resp, 'body') else resp
                symbol = getattr(order, 'symbol', '?')
                qty = getattr(order, 'qty', 0)
                side = getattr(order, 'side', '?')
                closed.append({"ticker": symbol, "qty": str(qty), "side": str(side)})
            return closed
        except Exception as e:
            log.error("Alpaca close_all_positions failed: %s", e)
            raise RuntimeError(f"Alpaca API error closing positions: {e}") from e

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as e:
            log.warning("Alpaca cancel_order failed for %s: %s", order_id, e)
            return False
