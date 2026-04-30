"""Execution bridge — connects analysis decisions to broker execution.

The complete pipeline:
  decisions → circuit breaker check → broker execution → audit logging

Supports three modes:
  - dry_run: Log decisions, no broker interaction (default, safe for testing)
  - paper:   Execute via Alpaca paper trading
  - live:    Execute via Alpaca live trading (requires explicit opt-in)

Usage:
    bridge = ExecutionBridge.paper()    # or .dry_run()
    result = bridge.execute(decisions, portfolio_snapshot, run_id)
"""

import time
from dataclasses import dataclass, field
from typing import Literal

from src.execution.base import BrokerBase, Order
from src.execution.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, PortfolioSnapshot
from src.execution.audit import get_audit_log
from src.utils.telegram import get_alerts
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Summary of an execution run."""
    mode: str
    orders: list[Order] = field(default_factory=list)
    blocked_orders: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return sum(1 for o in self.orders if o.status not in ("failed", "rejected", "blocked"))

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.orders if o.status in ("failed", "rejected"))

    @property
    def blocked_count(self) -> int:
        return len(self.blocked_orders)

    def summary(self) -> str:
        parts = [f"Mode: {self.mode}"]
        if self.orders:
            parts.append(f"Executed: {self.success_count}/{len(self.orders)}")
        if self.blocked_orders:
            parts.append(f"Blocked: {self.blocked_count}")
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        return " | ".join(parts)


class ExecutionBridge:
    """Bridge between portfolio decisions and broker execution with safety checks."""

    def __init__(
        self,
        mode: Literal["dry_run", "paper", "live"] = "dry_run",
        broker: BrokerBase | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.mode = mode
        self.broker = broker
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._audit = get_audit_log()
        self._alerts = get_alerts()

        if mode in ("paper", "live") and broker is None:
            raise ValueError(f"Broker required for mode={mode}")

    @classmethod
    def dry_run(cls) -> "ExecutionBridge":
        """Create a dry-run bridge (no real orders, full safety checks)."""
        return cls(mode="dry_run")

    @classmethod
    def paper(cls) -> "ExecutionBridge":
        """Create a paper-trading bridge using Alpaca."""
        from src.execution.alpaca import AlpacaBroker
        return cls(mode="paper", broker=AlpacaBroker())

    def execute(
        self,
        decisions: dict,
        portfolio_snapshot: PortfolioSnapshot | None = None,
        run_id: str = "",
    ) -> ExecutionResult:
        """Execute portfolio decisions with full safety pipeline.

        Args:
            decisions: {ticker: {action, quantity, confidence, reasoning, ...}}
            portfolio_snapshot: Current portfolio state for circuit breaker checks
            run_id: Correlation ID for audit trail

        Returns:
            ExecutionResult with order details, blocked orders, and errors.
        """
        result = ExecutionResult(mode=self.mode)

        # Build a default snapshot if none provided
        if portfolio_snapshot is None:
            portfolio_snapshot = PortfolioSnapshot(
                nav=100000, cash=100000, positions={},
                day_start_nav=100000, peak_nav=100000,
            )

        for ticker, decision in decisions.items():
            action = decision.get("action", "hold")
            quantity = decision.get("quantity", 0)
            confidence = decision.get("confidence") or 0
            reasoning = decision.get("reasoning", "")

            if action == "hold" or quantity <= 0:
                continue

            # Map action to broker side
            side_map = {"buy": "buy", "cover": "buy", "sell": "sell", "short": "sell"}
            side = side_map.get(action)
            if not side:
                continue

            # Get current price from decision or snapshot
            price = decision.get("current_price", 0)
            if price <= 0:
                # Try to get from portfolio snapshot
                pos = portfolio_snapshot.positions.get(ticker, {})
                price = pos.get("price", 0)

            # Circuit breaker check
            cb_ok, cb_reason = self.circuit_breaker.check_order(
                ticker=ticker,
                side=side,
                quantity=quantity,
                price=price if price > 0 else 1,  # Avoid division by zero
                portfolio=portfolio_snapshot,
                signal_confidence=confidence,
            )

            if not cb_ok:
                log.warning(
                    "Order BLOCKED by circuit breaker: %s %s %d — %s",
                    action, ticker, quantity, cb_reason,
                    agent_id="execution_bridge", ticker=ticker,
                )
                result.blocked_orders.append({
                    "ticker": ticker,
                    "action": action,
                    "quantity": quantity,
                    "reason": cb_reason,
                })
                self._audit.record_order(
                    run_id=run_id, ticker=ticker, side=side, quantity=quantity,
                    status="blocked", reasoning=reasoning,
                    circuit_breaker_check=cb_reason,
                )
                continue

            # Execute the order
            if self.mode == "dry_run":
                order = Order(
                    ticker=ticker, side=side, quantity=quantity,
                    status="simulated", reasoning=reasoning,
                )
                result.orders.append(order)
                log.info(
                    "DRY RUN: %s %d %s @ ~$%.2f",
                    side.upper(), quantity, ticker, price,
                    agent_id="execution_bridge", ticker=ticker,
                    action=side, quantity=quantity, mode="dry_run",
                )
                self._audit.record_order(
                    run_id=run_id, ticker=ticker, side=side, quantity=quantity,
                    status="simulated", reasoning=reasoning,
                    circuit_breaker_check="passed",
                )
            else:
                try:
                    order = self.broker.place_order(
                        ticker=ticker, side=side, quantity=quantity,
                    )
                    order.reasoning = reasoning
                    result.orders.append(order)
                    log.info(
                        "ORDER PLACED: %s %d %s — status=%s order_id=%s",
                        side.upper(), quantity, ticker, order.status, order.order_id,
                        agent_id="execution_bridge", ticker=ticker,
                        action=side, quantity=quantity,
                        order_id=order.order_id, status=order.status,
                        mode=self.mode,
                    )
                    # Normalize enum status to string for audit storage
                    status_str = str(order.status)
                    if "." in status_str:
                        status_str = status_str.split(".")[-1].lower()
                    self._audit.record_order(
                        run_id=run_id, ticker=ticker, side=side, quantity=quantity,
                        status=status_str, order_id=order.order_id,
                        filled_price=order.filled_price, filled_at=order.filled_at,
                        reasoning=reasoning, circuit_breaker_check="passed",
                    )
                    # Send Telegram alert for real trades (non-fatal)
                    try:
                        self._alerts.send_trade(
                            ticker=ticker, side=side, quantity=quantity,
                            price=order.filled_price or price, status=order.status,
                        )
                    except Exception as alert_err:
                        log.warning(
                            "Telegram alert failed (non-fatal): %s", alert_err,
                            ticker=ticker,
                        )
                except Exception as e:
                    error_msg = f"{ticker}: {e}"
                    result.errors.append(error_msg)
                    result.orders.append(Order(
                        ticker=ticker, side=side, quantity=quantity,
                        status="failed", reasoning=str(e),
                    ))
                    log.error(
                        "ORDER FAILED: %s %d %s — %s",
                        side.upper(), quantity, ticker, e,
                        agent_id="execution_bridge", ticker=ticker,
                    )
                    self._audit.record_order(
                        run_id=run_id, ticker=ticker, side=side, quantity=quantity,
                        status="failed", reasoning=str(e),
                        circuit_breaker_check="passed",
                    )

        log.info("Execution complete: %s", result.summary(), mode=self.mode)
        return result
