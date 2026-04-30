"""Persistent trade audit logging.

Every decision, order, and execution event is recorded in an append-only
SQLite database. This provides an immutable audit trail for:
  - Compliance review
  - Agent performance tracking
  - Debugging unexpected behavior
  - Post-trade analysis

Schema:
  - runs: One row per pipeline execution (run_id, tickers, model, timestamps)
  - signals: One row per agent signal per ticker (agent_id, ticker, signal, confidence, status)
  - decisions: One row per portfolio decision per ticker (action, quantity, reasoning, vote details)
  - orders: One row per executed order (order_id, status, filled_price, etc.)
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger, get_run_id

log = get_logger(__name__)

_DB_PATH = Path.home() / ".nova-trader" / "audit" / "trades.db"


class AuditLog:
    """Append-only trade audit database."""

    def __init__(self, db_path: str | None = None):
        self._db_path = str(db_path or _DB_PATH)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()
        # Auto-purge stale runs on startup
        self.mark_stale_runs_failed()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._get_conn() as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                os.chmod(self._db_path, 0o600)
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        started_at REAL NOT NULL,
                        completed_at REAL,
                        tickers TEXT NOT NULL,
                        model_name TEXT,
                        model_provider TEXT,
                        mode TEXT DEFAULT 'dry_run',
                        status TEXT DEFAULT 'running',
                        error TEXT
                    );

                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        signal TEXT,
                        confidence REAL,
                        status TEXT DEFAULT 'success',
                        reasoning TEXT,
                        recorded_at REAL NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    );

                    CREATE TABLE IF NOT EXISTS decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        action TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        confidence REAL,
                        reasoning TEXT,
                        weighted_score REAL,
                        vote_details TEXT,
                        recorded_at REAL NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    );

                    CREATE TABLE IF NOT EXISTS orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        order_id TEXT,
                        ticker TEXT NOT NULL,
                        side TEXT NOT NULL,
                        quantity INTEGER NOT NULL,
                        order_type TEXT DEFAULT 'market',
                        status TEXT NOT NULL,
                        filled_price REAL,
                        filled_at TEXT,
                        reasoning TEXT,
                        circuit_breaker_check TEXT,
                        recorded_at REAL NOT NULL,
                        FOREIGN KEY (run_id) REFERENCES runs(run_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_signals_run ON signals(run_id);
                    CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);
                    CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
                    CREATE INDEX IF NOT EXISTS idx_orders_run ON orders(run_id);
                    CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
                """)
        except Exception as e:
            log.error("Failed to initialize audit database: %s", e)

    def record_run_start(
        self,
        run_id: str,
        tickers: list[str],
        model_name: str = "",
        model_provider: str = "",
        mode: str = "dry_run",
    ) -> None:
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO runs (run_id, started_at, tickers, model_name, model_provider, mode) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, time.time(), json.dumps(tickers), model_name, model_provider, mode),
                )
                conn.commit()
            except Exception as e:
                log.error("Failed to record run start: %s", e)

    def update_run_mode(self, run_id: str, mode: str) -> None:
        """Update the execution mode for a run (e.g. from dry_run to paper after execution)."""
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute("UPDATE runs SET mode=? WHERE run_id=?", (mode, run_id))
                conn.commit()
            except Exception as e:
                log.error("Failed to update run mode: %s", e)

    def record_run_complete(self, run_id: str, status: str = "completed", error: str | None = None) -> None:
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    "UPDATE runs SET completed_at=?, status=?, error=? WHERE run_id=?",
                    (time.time(), status, error, run_id),
                )
                conn.commit()
            except Exception as e:
                log.error("Failed to record run completion: %s", e)

    def record_signal(
        self,
        run_id: str,
        agent_id: str,
        ticker: str,
        signal: str | None,
        confidence: float,
        status: str = "success",
        reasoning: str | None = None,
    ) -> None:
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO signals (run_id, agent_id, ticker, signal, confidence, status, reasoning, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, agent_id, ticker, signal, confidence, status, reasoning, time.time()),
                )
                conn.commit()
            except Exception as e:
                log.error("Failed to record signal: %s", e)

    def record_signals_batch(self, run_id: str, analyst_signals: dict) -> None:
        """Record all analyst signals from state["data"]["analyst_signals"]."""
        for agent_id, ticker_signals in analyst_signals.items():
            if agent_id.startswith("risk_management_agent"):
                continue
            for ticker, sig in ticker_signals.items():
                if isinstance(sig, dict):
                    self.record_signal(
                        run_id=run_id,
                        agent_id=agent_id,
                        ticker=ticker,
                        signal=sig.get("signal"),
                        confidence=sig.get("confidence", 0),
                        status=sig.get("status", "success"),
                        reasoning=json.dumps(sig.get("reasoning")) if sig.get("reasoning") else None,
                    )

    def record_decision(
        self,
        run_id: str,
        ticker: str,
        action: str,
        quantity: int,
        confidence: float = 0,
        reasoning: str = "",
        weighted_score: float = 0,
        vote_details: dict | None = None,
    ) -> None:
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO decisions (run_id, ticker, action, quantity, confidence, reasoning, weighted_score, vote_details, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, ticker, action, quantity, confidence, reasoning, weighted_score, json.dumps(vote_details) if vote_details else None, time.time()),
                )
                conn.commit()
            except Exception as e:
                log.error("Failed to record decision: %s", e)

    def record_order(
        self,
        run_id: str,
        ticker: str,
        side: str,
        quantity: int,
        status: str,
        order_id: str | None = None,
        order_type: str = "market",
        filled_price: float | None = None,
        filled_at: str | None = None,
        reasoning: str = "",
        circuit_breaker_check: str = "",
    ) -> None:
        with self._lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    "INSERT INTO orders (run_id, order_id, ticker, side, quantity, order_type, status, filled_price, filled_at, reasoning, circuit_breaker_check, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (run_id, order_id, ticker, side, quantity, order_type, status, filled_price, filled_at, reasoning, circuit_breaker_check, time.time()),
                )
                conn.commit()
            except Exception as e:
                log.error("Failed to record order: %s", e)

    def mark_stale_runs_failed(self) -> int:
        """Mark runs stuck in 'running' for over 1 hour as failed."""
        cutoff = time.time() - 3600  # 1 hour ago
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.execute(
                        "UPDATE runs SET status='failed', error='timeout: stale run cleaned up' "
                        "WHERE status='running' AND started_at < ?",
                        (cutoff,),
                    )
                    count = cursor.rowcount
                    if count:
                        log.info("Cleaned %d stale runs", count)
                    return count
            except Exception as e:
                log.error("Failed to clean stale runs: %s", e)
                return 0

    def clean_stale(self) -> str:
        """Clean stale runs and return a human-readable summary."""
        count = self.mark_stale_runs_failed()
        if count:
            return f"Cleaned {count} stale run{'s' if count != 1 else ''}"
        return "No stale runs found"

    def get_recent_runs(self, limit: int = 20) -> list[dict]:
        """Fetch recent pipeline runs."""
        try:
            with self._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_recent_runs_with_summary(self, limit: int = 20) -> list[dict]:
        """Fetch recent runs with decision and order summary counts.

        Returns each run dict augmented with:
          - buy_count, sell_count, hold_count (from decisions)
          - total_decisions (total decision rows)
          - executed_orders (orders with status 'filled' or 'executed')
          - total_orders (all order rows)
        """
        try:
            with self._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        r.*,
                        COALESCE(d.buy_count, 0)   AS buy_count,
                        COALESCE(d.sell_count, 0)   AS sell_count,
                        COALESCE(d.hold_count, 0)   AS hold_count,
                        COALESCE(d.total_decisions, 0) AS total_decisions,
                        COALESCE(o.executed_orders, 0) AS executed_orders,
                        COALESCE(o.total_orders, 0)    AS total_orders
                    FROM runs r
                    LEFT JOIN (
                        SELECT
                            run_id,
                            SUM(CASE WHEN LOWER(action) IN ('buy', 'strong_buy', 'cover') THEN 1 ELSE 0 END) AS buy_count,
                            SUM(CASE WHEN LOWER(action) IN ('sell', 'strong_sell', 'short') THEN 1 ELSE 0 END) AS sell_count,
                            SUM(CASE WHEN LOWER(action) IN ('hold') THEN 1 ELSE 0 END) AS hold_count,
                            COUNT(*) AS total_decisions
                        FROM decisions
                        GROUP BY run_id
                    ) d ON d.run_id = r.run_id
                    LEFT JOIN (
                        SELECT
                            run_id,
                            SUM(CASE WHEN LOWER(status) IN ('filled', 'executed', 'pending_new', 'accepted', 'new') OR LOWER(status) LIKE '%pending_new%' THEN 1 ELSE 0 END) AS executed_orders,
                            COUNT(*) AS total_orders
                        FROM orders
                        GROUP BY run_id
                    ) o ON o.run_id = r.run_id
                    ORDER BY r.started_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_run_details(self, run_id: str) -> dict:
        """Get full details for a specific run."""
        try:
            with self._get_conn() as conn:
                conn.row_factory = sqlite3.Row
                run = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                signals = conn.execute("SELECT * FROM signals WHERE run_id=? ORDER BY ticker, agent_id", (run_id,)).fetchall()
                decisions = conn.execute("SELECT * FROM decisions WHERE run_id=? ORDER BY ticker", (run_id,)).fetchall()
                orders = conn.execute("SELECT * FROM orders WHERE run_id=? ORDER BY recorded_at", (run_id,)).fetchall()
                return {
                    "run": dict(run) if run else None,
                    "signals": [dict(s) for s in signals],
                    "decisions": [dict(d) for d in decisions],
                    "orders": [dict(o) for o in orders],
                }
        except Exception:
            return {}


# Global singleton (thread-safe lazy init)
_audit_log: AuditLog | None = None
_audit_log_lock = threading.Lock()


def get_audit_log() -> AuditLog:
    global _audit_log
    if _audit_log is None:
        with _audit_log_lock:
            if _audit_log is None:
                _audit_log = AuditLog()
    return _audit_log
