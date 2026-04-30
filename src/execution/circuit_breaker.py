"""Circuit breakers and kill switch for trade execution safety.

Prevents catastrophic losses by enforcing hard limits that cannot be
overridden by any agent or LLM decision.

Features:
  - Global kill switch (file-based — touch ~/.nova-trader/KILL_SWITCH to halt)
  - Max daily loss limit (default: -2% of starting NAV)
  - Max drawdown from peak (default: -10%)
  - Per-trade notional value cap (default: 5% of NAV)
  - Per-ticker position limit as % of portfolio (default: 20%)
  - Order rate limiter (max N orders per minute)
  - Minimum signal health check (>50% agents must succeed)
"""

import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

KILL_SWITCH_PATH = Path.home() / ".nova-trader" / "KILL_SWITCH"
CIRCUIT_TRIPPED_PATH = Path.home() / ".nova-trader" / "CIRCUIT_TRIPPED"


@dataclass
class CircuitBreakerConfig:
    """All configurable limits. Conservative defaults for safety."""
    max_daily_loss_pct: float = 0.02        # 2% max daily loss
    max_drawdown_pct: float = 0.10          # 10% max drawdown from peak
    max_trade_notional_pct: float = 0.05    # 5% of NAV per single trade
    max_position_pct: float = 0.20          # 20% of NAV per ticker
    max_orders_per_minute: int = 10         # Rate limit
    min_signal_health_pct: float = 0.50     # At least 50% agents must succeed
    max_total_gross_exposure_pct: float = 2.0  # 200% gross exposure cap
    min_confidence_to_trade: float = 60.0   # Minimum confidence score to execute


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state for circuit breaker checks."""
    nav: float  # Net Asset Value (total portfolio value)
    cash: float
    positions: dict[str, dict]  # {ticker: {long, short, value}}
    day_start_nav: float  # NAV at start of trading day
    peak_nav: float  # Highest NAV ever recorded


class CircuitBreaker:
    """Validates every order against hard safety limits.

    Usage:
        cb = CircuitBreaker()
        # Before executing any order:
        ok, reason = cb.check_order(order, portfolio_snapshot)
        if not ok:
            log.warning("Order blocked: %s", reason)
            return
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.Lock()
        self._order_timestamps: list[float] = []
        self._tripped = False
        self._trip_reason = ""
        # Restore persisted trip state from previous run
        self._restore_trip_state()

    # ── Kill Switch ──────────────────────────────────────

    @staticmethod
    def is_killed() -> bool:
        """Check if the global kill switch is active."""
        return KILL_SWITCH_PATH.exists()

    @staticmethod
    def activate_kill_switch(reason: str = "Manual activation") -> None:
        """Activate the kill switch. Creates the sentinel file."""
        KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        KILL_SWITCH_PATH.write_text(f"HALTED: {reason}\nTime: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        log.critical("KILL SWITCH ACTIVATED: %s", reason)

    @staticmethod
    def deactivate_kill_switch() -> None:
        """Deactivate the kill switch. Removes the sentinel file."""
        if KILL_SWITCH_PATH.exists():
            KILL_SWITCH_PATH.unlink()
            log.info("Kill switch deactivated")

    # ── Core Check ───────────────────────────────────────

    def check_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float,
        portfolio: PortfolioSnapshot,
        signal_confidence: float = 100.0,
    ) -> tuple[bool, str]:
        """Validate an order against all circuit breaker rules.

        Returns:
            (True, "") if order is allowed
            (False, "reason") if order is blocked
        """
        # Kill switch — absolute priority
        if self.is_killed():
            return False, "KILL SWITCH ACTIVE — all trading halted"

        # Already tripped this session (read under lock for thread safety)
        with self._lock:
            if self._tripped:
                return False, f"Circuit breaker tripped: {self._trip_reason}"

        # 1. Basic sanity checks FIRST — before consuming rate limit slots
        if quantity <= 0:
            return False, "Invalid quantity: must be positive"
        if price <= 0:
            return False, "Invalid price: must be positive"

        notional = quantity * price

        if notional < 1:
            return False, "Trade too small: notional < $1"

        # 2. Daily loss check
        daily_pnl_pct = (portfolio.nav - portfolio.day_start_nav) / portfolio.day_start_nav if portfolio.day_start_nav > 0 else 0
        if daily_pnl_pct < -self.config.max_daily_loss_pct:
            self._trip("max_daily_loss", f"Daily loss {daily_pnl_pct:.2%} exceeds limit {-self.config.max_daily_loss_pct:.2%}")
            with self._lock:
                return False, self._trip_reason

        # 3. Max drawdown check
        drawdown = (portfolio.nav - portfolio.peak_nav) / portfolio.peak_nav if portfolio.peak_nav > 0 else 0
        if drawdown < -self.config.max_drawdown_pct:
            self._trip("max_drawdown", f"Drawdown {drawdown:.2%} exceeds limit {-self.config.max_drawdown_pct:.2%}")
            with self._lock:
                return False, self._trip_reason

        # 4. Per-trade notional cap
        max_trade_notional = portfolio.nav * self.config.max_trade_notional_pct
        if notional > max_trade_notional:
            return False, f"Trade notional ${notional:,.0f} exceeds {self.config.max_trade_notional_pct:.0%} of NAV (${max_trade_notional:,.0f})"

        # 5. Per-ticker position limit
        pos = portfolio.positions.get(ticker, {})
        existing_value = abs(pos.get("value", 0))
        new_total = existing_value + notional
        max_position = portfolio.nav * self.config.max_position_pct
        if new_total > max_position:
            return False, f"Position in {ticker} would be ${new_total:,.0f}, exceeding {self.config.max_position_pct:.0%} limit (${max_position:,.0f})"

        # 6. Gross exposure cap
        total_gross = sum(abs(p.get("value", 0)) for p in portfolio.positions.values())
        new_gross = total_gross + notional
        max_gross = portfolio.nav * self.config.max_total_gross_exposure_pct
        if new_gross > max_gross:
            return False, f"Gross exposure ${new_gross:,.0f} would exceed {self.config.max_total_gross_exposure_pct:.0%} cap (${max_gross:,.0f})"

        # 7. Minimum confidence
        if signal_confidence < self.config.min_confidence_to_trade:
            return False, f"Signal confidence {signal_confidence:.0f}% below minimum {self.config.min_confidence_to_trade:.0f}%"

        # 8. Rate limiter — LAST, so invalid orders don't consume slots
        if not self._check_rate_limit():
            return False, f"Rate limit exceeded: max {self.config.max_orders_per_minute} orders/minute"

        return True, ""

    def check_signal_health(self, analyst_signals: dict) -> tuple[bool, str]:
        """Check if enough agents succeeded to trust the signals.

        Should be called BEFORE generating any orders.
        """
        from src.core.signals import count_signal_health
        health = count_signal_health(analyst_signals)

        if health["total"] == 0:
            return False, "No agent signals received"

        failure_rate = health["failed"] / health["total"]
        if failure_rate > (1 - self.config.min_signal_health_pct):
            return False, f"Too many agent failures: {health['failed']}/{health['total']} ({failure_rate:.0%}) — minimum {self.config.min_signal_health_pct:.0%} must succeed"

        if health["failed"] > 0:
            log.warning(
                "Degraded signal health: %d/%d agents failed",
                health["failed"], health["total"],
            )

        return True, ""

    # ── Internal ─────────────────────────────────────────

    def _restore_trip_state(self) -> None:
        """Restore tripped state from persistent file if it exists."""
        try:
            if CIRCUIT_TRIPPED_PATH.exists():
                reason = CIRCUIT_TRIPPED_PATH.read_text().strip()
                self._tripped = True
                self._trip_reason = reason or "Restored from previous session"
                log.warning("Circuit breaker restored tripped state: %s", self._trip_reason)
        except Exception as e:
            log.warning("Could not restore circuit breaker state: %s", e)

    def _trip(self, breaker_name: str, reason: str) -> None:
        """Trip a circuit breaker — halts all trading for this session."""
        with self._lock:
            self._tripped = True
            self._trip_reason = reason
        # Persist trip state to survive restarts
        try:
            CIRCUIT_TRIPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
            CIRCUIT_TRIPPED_PATH.write_text(f"[{breaker_name}] {reason}")
        except Exception as e:
            log.warning("Could not persist circuit breaker state: %s", e)
        log.critical("CIRCUIT BREAKER TRIPPED [%s]: %s", breaker_name, reason)

    def _check_rate_limit(self) -> bool:
        """Check and update order rate limiter."""
        now = time.time()
        with self._lock:
            # Remove timestamps older than 60 seconds
            self._order_timestamps = [ts for ts in self._order_timestamps if now - ts < 60]
            if len(self._order_timestamps) >= self.config.max_orders_per_minute:
                return False
            self._order_timestamps.append(now)
            return True

    def reset(self) -> None:
        """Reset the circuit breaker state (e.g., for a new trading day).

        Also removes the persistent trip file so the state does not
        survive the next restart.
        """
        with self._lock:
            self._tripped = False
            self._trip_reason = ""
            self._order_timestamps.clear()
        CIRCUIT_TRIPPED_PATH.unlink(missing_ok=True)
        log.info("Circuit breaker reset")

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            return self._tripped

    @property
    def trip_reason(self) -> str:
        with self._lock:
            return self._trip_reason
