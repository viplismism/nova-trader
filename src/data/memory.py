"""Trading memory system for learning from past decisions.

Persistent storage of market events, trade outcomes, agent accuracy,
and portfolio snapshots. This gives the system institutional memory —
it can recall what happened last time it saw similar signals, which
agents have been most accurate, and how the portfolio has performed.

Tables:
  - market_events: Significant events (earnings, fed decisions, crashes, etc.)
  - trade_outcomes: What happened after each trade (P&L, lessons learned)
  - agent_accuracy: Per-agent prediction tracking over time
  - portfolio_snapshots: Daily NAV and position state
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path.home() / ".nova-trader" / "memory" / "trading_memory.db"


class TradingMemory:
    """Persistent trading memory backed by SQLite."""

    def __init__(self, db_path: str | None = None):
        self._db_path = str(db_path or _DB_PATH)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                os.chmod(self._db_path, 0o600)
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS market_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        tickers TEXT NOT NULL,
                        description TEXT NOT NULL,
                        impact TEXT NOT NULL,
                        magnitude INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS trade_outcomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        action TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        exit_price REAL,
                        pnl_pct REAL,
                        holding_period_days INTEGER,
                        signal_at_entry TEXT,
                        outcome TEXT,
                        lessons_learned TEXT,
                        recorded_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS agent_accuracy (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        agent_id TEXT NOT NULL,
                        ticker TEXT NOT NULL,
                        date TEXT NOT NULL,
                        predicted_signal TEXT NOT NULL,
                        actual_outcome TEXT,
                        was_correct INTEGER,
                        confidence_at_prediction REAL,
                        recorded_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        nav REAL NOT NULL,
                        cash REAL NOT NULL,
                        positions_json TEXT,
                        daily_return_pct REAL,
                        cumulative_return_pct REAL,
                        recorded_at REAL NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_type ON market_events(event_type);
                    CREATE INDEX IF NOT EXISTS idx_events_tickers ON market_events(tickers);
                    CREATE INDEX IF NOT EXISTS idx_outcomes_ticker ON trade_outcomes(ticker);
                    CREATE INDEX IF NOT EXISTS idx_outcomes_run ON trade_outcomes(run_id);
                    CREATE INDEX IF NOT EXISTS idx_accuracy_agent ON agent_accuracy(agent_id);
                    CREATE INDEX IF NOT EXISTS idx_accuracy_ticker ON agent_accuracy(ticker);
                    CREATE INDEX IF NOT EXISTS idx_accuracy_date ON agent_accuracy(date);
                    CREATE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(date);
                """)
        except Exception as e:
            log.error("Failed to initialize trading memory database: %s", e)

    # ── Recording methods ──

    def record_market_event(
        self,
        event_type: str,
        tickers: list[str],
        description: str,
        impact: str,
        magnitude: int,
    ) -> None:
        """Record a significant market event.

        Args:
            event_type: One of earnings, fed_decision, crash, rally, news.
            tickers: List of affected tickers.
            description: Human-readable description of the event.
            impact: One of bullish, bearish, neutral.
            magnitude: Impact severity from 0-100.
        """
        with self._lock:
            try:
                with sqlite3.connect(self._db_path, timeout=10) as conn:
                    conn.execute(
                        "INSERT INTO market_events (timestamp, event_type, tickers, description, impact, magnitude) VALUES (?, ?, ?, ?, ?, ?)",
                        (time.time(), event_type, ",".join(tickers), description, impact, magnitude),
                    )
                log.info(
                    "Recorded market event: %s (%s)",
                    event_type,
                    ",".join(tickers),
                )
            except Exception as e:
                log.error("Failed to record market event: %s", e)

    def record_trade_outcome(
        self,
        run_id: str,
        ticker: str,
        action: str,
        entry_price: float,
        exit_price: float | None = None,
        signal_at_entry: dict | None = None,
        lessons_learned: str | None = None,
    ) -> None:
        """Record the outcome of a trade.

        P&L, outcome, and holding period are computed automatically when
        exit_price is provided.
        """
        pnl_pct = None
        outcome = None
        if exit_price is not None and entry_price > 0:
            if action == "buy":
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            elif action == "sell":
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100

            if pnl_pct is not None:
                if pnl_pct > 0.5:
                    outcome = "win"
                elif pnl_pct < -0.5:
                    outcome = "loss"
                else:
                    outcome = "flat"

        with self._lock:
            try:
                with sqlite3.connect(self._db_path, timeout=10) as conn:
                    conn.execute(
                        "INSERT INTO trade_outcomes (run_id, ticker, action, entry_price, exit_price, pnl_pct, holding_period_days, signal_at_entry, outcome, lessons_learned, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            run_id,
                            ticker,
                            action,
                            entry_price,
                            exit_price,
                            pnl_pct,
                            None,  # holding_period_days — set later when trade closes
                            json.dumps(signal_at_entry) if signal_at_entry else None,
                            outcome,
                            lessons_learned,
                            time.time(),
                        ),
                    )
                log.info(
                    "Recorded trade outcome: %s %s @ %.2f -> %s",
                    action,
                    ticker,
                    entry_price,
                    outcome or "open",
                    ticker=ticker,
                    action=action,
                )
            except Exception as e:
                log.error("Failed to record trade outcome: %s", e)

    def record_agent_accuracy(
        self,
        agent_id: str,
        ticker: str,
        predicted_signal: str,
        actual_outcome: str,
        confidence: float,
    ) -> None:
        """Record whether an agent's prediction was correct.

        Args:
            agent_id: The agent that made the prediction.
            ticker: Ticker that was predicted on.
            predicted_signal: One of bullish, bearish, neutral.
            actual_outcome: One of up, down, flat.
            confidence: Agent's confidence at time of prediction (0-1).
        """
        was_correct = (
            (predicted_signal == "bullish" and actual_outcome == "up")
            or (predicted_signal == "bearish" and actual_outcome == "down")
            or (predicted_signal == "neutral" and actual_outcome == "flat")
        )
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            try:
                with sqlite3.connect(self._db_path, timeout=10) as conn:
                    conn.execute(
                        "INSERT INTO agent_accuracy (agent_id, ticker, date, predicted_signal, actual_outcome, was_correct, confidence_at_prediction, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (agent_id, ticker, today, predicted_signal, actual_outcome, int(was_correct), confidence, time.time()),
                    )
                log.info(
                    "Recorded agent accuracy: %s on %s — %s (correct: %s)",
                    agent_id,
                    ticker,
                    predicted_signal,
                    was_correct,
                    agent_id=agent_id,
                    ticker=ticker,
                )
            except Exception as e:
                log.error("Failed to record agent accuracy: %s", e)

    def record_portfolio_snapshot(
        self,
        nav: float,
        cash: float,
        positions: dict,
        daily_return_pct: float,
    ) -> None:
        """Record a daily portfolio snapshot."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            try:
                with sqlite3.connect(self._db_path, timeout=10) as conn:
                    # Compute cumulative return from first snapshot (inside lock
                    # so the read + insert is atomic with respect to other writers)
                    cumulative = None
                    row = conn.execute(
                        "SELECT nav FROM portfolio_snapshots ORDER BY date ASC LIMIT 1"
                    ).fetchone()
                    if row and row[0] > 0:
                        cumulative = ((nav - row[0]) / row[0]) * 100

                    conn.execute(
                        "INSERT INTO portfolio_snapshots (date, nav, cash, positions_json, daily_return_pct, cumulative_return_pct, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (today, nav, cash, json.dumps(positions, default=str), daily_return_pct, cumulative, time.time()),
                    )
                log.info("Recorded portfolio snapshot: NAV=%.2f, daily=%.2f%%", nav, daily_return_pct)
            except Exception as e:
                log.error("Failed to record portfolio snapshot: %s", e)

    def record_pending_prediction(
        self,
        agent_id: str,
        ticker: str,
        predicted_signal: str,
        confidence: float,
        price_at_prediction: float,
    ) -> None:
        """Record a pending agent prediction (outcome not yet known).

        Args:
            agent_id: The agent that made the prediction.
            ticker: Ticker that was predicted on.
            predicted_signal: One of bullish, bearish, neutral.
            confidence: Agent's confidence at time of prediction (0-1).
            price_at_prediction: Price of the ticker when prediction was made.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._lock:
            try:
                with sqlite3.connect(self._db_path, timeout=10) as conn:
                    conn.execute(
                        "INSERT INTO agent_accuracy (agent_id, ticker, date, predicted_signal, actual_outcome, was_correct, confidence_at_prediction, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (agent_id, ticker, today, predicted_signal, "pending", 0, confidence, time.time()),
                    )
            except Exception as e:
                log.error("Failed to record pending prediction: %s", e)

    def update_prediction_outcome(
        self,
        agent_id: str,
        ticker: str,
        actual_outcome: str,
        was_correct: bool,
    ) -> bool:
        """Update the most recent pending prediction for an agent+ticker with the actual outcome.

        Args:
            agent_id: The agent that made the prediction.
            ticker: Ticker that was predicted on.
            actual_outcome: One of up, down, flat.
            was_correct: Whether the prediction direction matched.

        Returns:
            True if a pending record was updated, False otherwise.
        """
        with self._lock:
            try:
                with sqlite3.connect(self._db_path, timeout=10) as conn:
                    # Find the most recent pending record for this agent+ticker
                    row = conn.execute(
                        "SELECT id FROM agent_accuracy WHERE agent_id=? AND ticker=? AND actual_outcome='pending' ORDER BY recorded_at DESC LIMIT 1",
                        (agent_id, ticker),
                    ).fetchone()
                    if not row:
                        return False
                    conn.execute(
                        "UPDATE agent_accuracy SET actual_outcome=?, was_correct=? WHERE id=?",
                        (actual_outcome, int(was_correct), row[0]),
                    )
                    log.info(
                        "Updated prediction outcome: %s on %s — %s (correct: %s)",
                        agent_id, ticker, actual_outcome, was_correct,
                        agent_id=agent_id, ticker=ticker,
                    )
                    return True
            except Exception as e:
                log.error("Failed to update prediction outcome: %s", e)
                return False

    # ── Query methods ──

    def get_ticker_history(self, ticker: str, limit: int = 20) -> list[dict]:
        """Get past trades on a specific ticker."""
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM trade_outcomes WHERE ticker=? ORDER BY recorded_at DESC LIMIT ?",
                    (ticker, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_agent_accuracy(self, agent_id: str, lookback_days: int = 90) -> dict:
        """Get accuracy stats for a specific agent over a lookback period.

        Returns:
            Dict with total, correct, accuracy_pct, avg_confidence.
        """
        cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT was_correct, confidence_at_prediction FROM agent_accuracy WHERE agent_id=? AND date >= date(?, ?) AND actual_outcome != 'pending'",
                    (agent_id, cutoff, f"-{lookback_days} days"),
                ).fetchall()

                if not rows:
                    return {"total": 0, "correct": 0, "accuracy_pct": 0.0, "avg_confidence": 0.0}

                total = len(rows)
                correct = sum(1 for r in rows if r["was_correct"])
                avg_conf = sum(r["confidence_at_prediction"] for r in rows) / total

                return {
                    "total": total,
                    "correct": correct,
                    "accuracy_pct": (correct / total) * 100 if total > 0 else 0.0,
                    "avg_confidence": round(avg_conf, 3),
                }
        except Exception:
            return {"total": 0, "correct": 0, "accuracy_pct": 0.0, "avg_confidence": 0.0}

    def get_best_agents(self, lookback_days: int = 90) -> list[dict]:
        """Get agents ranked by accuracy over a lookback period.

        Returns:
            List of dicts with agent_id, total, correct, accuracy_pct, avg_confidence,
            sorted by accuracy descending.
        """
        cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        agent_id,
                        COUNT(*) as total,
                        SUM(was_correct) as correct,
                        ROUND(CAST(SUM(was_correct) AS REAL) / COUNT(*) * 100, 1) as accuracy_pct,
                        ROUND(AVG(confidence_at_prediction), 3) as avg_confidence
                    FROM agent_accuracy
                    WHERE date >= date(?, ?) AND actual_outcome != 'pending'
                    GROUP BY agent_id
                    HAVING total >= 5
                    ORDER BY accuracy_pct DESC
                    """,
                    (cutoff, f"-{lookback_days} days"),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []

    def get_similar_situations(self, ticker: str, current_signals: dict) -> list[dict]:
        """Find past trades with similar signal patterns.

        Compares current agent signals against signal_at_entry blobs
        from previous trades on the same ticker. Returns trades where
        at least half the agents agreed on the same direction.

        Args:
            ticker: The ticker to search for.
            current_signals: Dict of agent_id -> signal (bullish/bearish/neutral).

        Returns:
            List of matching past trade outcome dicts.
        """
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM trade_outcomes WHERE ticker=? AND signal_at_entry IS NOT NULL ORDER BY recorded_at DESC LIMIT 100",
                    (ticker,),
                ).fetchall()

                if not rows or not current_signals:
                    return []

                # Score each past trade by signal similarity
                matches = []
                for row in rows:
                    row_dict = dict(row)
                    try:
                        past_signals = json.loads(row_dict["signal_at_entry"])
                    except (json.JSONDecodeError, TypeError):
                        continue

                    # Count how many agents gave the same signal
                    overlap = 0
                    compared = 0
                    for agent_id, current_sig in current_signals.items():
                        if agent_id in past_signals:
                            compared += 1
                            past_sig = past_signals[agent_id]
                            # Handle both string signals and dict signals
                            if isinstance(past_sig, dict):
                                past_sig = past_sig.get("signal", "")
                            if isinstance(current_sig, dict):
                                current_sig = current_sig.get("signal", "")
                            if past_sig == current_sig:
                                overlap += 1

                    if compared > 0 and overlap / compared >= 0.5:
                        row_dict["_similarity_score"] = overlap / compared
                        matches.append(row_dict)

                # Sort by similarity, most similar first
                matches.sort(key=lambda x: x["_similarity_score"], reverse=True)
                return matches[:10]

        except Exception:
            return []

    def get_portfolio_performance(self, days: int = 30) -> list[dict]:
        """Get daily NAV series for the last N days."""
        cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self._db_path, timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM portfolio_snapshots WHERE date >= date(?, ?) ORDER BY date ASC",
                    (cutoff, f"-{days} days"),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []


# Global singleton (thread-safe lazy init)
_trading_memory: TradingMemory | None = None
_trading_memory_lock = threading.Lock()


def get_trading_memory() -> TradingMemory:
    """Get the global TradingMemory singleton."""
    global _trading_memory
    if _trading_memory is None:
        with _trading_memory_lock:
            if _trading_memory is None:
                _trading_memory = TradingMemory()
    return _trading_memory
