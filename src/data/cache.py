"""Thread-safe cache with TTL and SQLite persistence.

Replaces the old in-memory-only cache. Features:
  - Thread-safe (all operations protected by RLock)
  - TTL per data type (prices: 5min, metrics: 1hr, filings: 24hr)
  - SQLite persistence (survives restarts, prevents API hammering)
  - Data freshness tracking (each entry has a stored_at timestamp)
  - Memory layer on top of SQLite for fast repeated access
"""

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger(__name__)

# TTL in seconds per data type
TTL_PRICES = 300           # 5 minutes
TTL_FINANCIAL_METRICS = 3600  # 1 hour
TTL_LINE_ITEMS = 3600      # 1 hour
TTL_INSIDER_TRADES = 86400  # 24 hours
TTL_COMPANY_NEWS = 1800     # 30 minutes

_TTL_MAP = {
    "prices": TTL_PRICES,
    "financial_metrics": TTL_FINANCIAL_METRICS,
    "line_items": TTL_LINE_ITEMS,
    "insider_trades": TTL_INSIDER_TRADES,
    "company_news": TTL_COMPANY_NEWS,
}


class Cache:
    """Thread-safe cache with TTL and optional SQLite persistence."""

    def __init__(self, db_path: str | None = None):
        self._lock = threading.RLock()
        self._mem: dict[str, dict[str, tuple[float, list[dict]]]] = {
            "prices": {},
            "financial_metrics": {},
            "line_items": {},
            "insider_trades": {},
            "company_news": {},
        }

        # SQLite persistence — use thread-local connections to avoid FD churn
        self._db_path = db_path or self._default_db_path()
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection."""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            self._local.conn = conn
        return conn

    @staticmethod
    def _default_db_path() -> str:
        cache_dir = Path.home() / ".nova-trader" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return str(cache_dir / "data_cache.db")

    def _init_db(self) -> None:
        """Create the SQLite cache table if it doesn't exist."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                os.chmod(self._db_path, 0o600)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        data_type TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        data TEXT NOT NULL,
                        stored_at REAL NOT NULL,
                        PRIMARY KEY (data_type, cache_key)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_stored ON cache(stored_at)")
                conn.commit()
        except Exception as e:
            log.warning("SQLite cache init failed, using memory only: %s", e)

    def _is_fresh(self, data_type: str, stored_at: float) -> bool:
        """Check if a cached entry is still within TTL."""
        ttl = _TTL_MAP.get(data_type, 3600)
        return (time.time() - stored_at) < ttl

    def get(self, data_type: str, cache_key: str) -> list[dict] | None:
        """Get cached data if fresh. Checks memory first, then SQLite."""
        with self._lock:
            # Check memory
            bucket = self._mem.get(data_type, {})
            if cache_key in bucket:
                stored_at, data = bucket[cache_key]
                if self._is_fresh(data_type, stored_at):
                    return data
                else:
                    del bucket[cache_key]

            # Check SQLite
            try:
                conn = self._get_conn()
                row = conn.execute(
                    "SELECT data, stored_at FROM cache WHERE data_type=? AND cache_key=?",
                    (data_type, cache_key),
                ).fetchone()
                if row:
                    data_str, stored_at = row
                    if self._is_fresh(data_type, stored_at):
                        data = json.loads(data_str)
                        # Promote to memory
                        bucket[cache_key] = (stored_at, data)
                        return data
                    else:
                        conn.execute(
                            "DELETE FROM cache WHERE data_type=? AND cache_key=?",
                            (data_type, cache_key),
                        )
                        conn.commit()
            except Exception:
                pass

        return None

    def set(self, data_type: str, cache_key: str, data: list[dict]) -> None:
        """Store data in both memory and SQLite."""
        now = time.time()
        with self._lock:
            # Memory
            if data_type not in self._mem:
                self._mem[data_type] = {}
            self._mem[data_type][cache_key] = (now, data)

            # SQLite
            try:
                data_str = json.dumps(data, default=str)
                conn = self._get_conn()
                conn.execute(
                    "INSERT OR REPLACE INTO cache (data_type, cache_key, data, stored_at) VALUES (?, ?, ?, ?)",
                    (data_type, cache_key, data_str, now),
                )
                conn.commit()
            except Exception as e:
                log.warning("SQLite cache write failed: %s", e)

    def clear(self, data_type: str | None = None) -> None:
        """Clear cache. If data_type given, only clear that type."""
        with self._lock:
            if data_type:
                self._mem[data_type] = {}
                try:
                    conn = self._get_conn()
                    conn.execute("DELETE FROM cache WHERE data_type=?", (data_type,))
                    conn.commit()
                except Exception:
                    pass
            else:
                for key in self._mem:
                    self._mem[key] = {}
                try:
                    conn = self._get_conn()
                    conn.execute("DELETE FROM cache")
                    conn.commit()
                except Exception:
                    pass

    def stats(self) -> dict[str, int]:
        """Return count of entries per data type in memory."""
        with self._lock:
            return {k: len(v) for k, v in self._mem.items()}

    # ── Convenience methods (backward-compatible with old Cache API) ──

    def get_prices(self, cache_key: str) -> list[dict] | None:
        return self.get("prices", cache_key)

    def set_prices(self, cache_key: str, data: list[dict]) -> None:
        self.set("prices", cache_key, data)

    def get_financial_metrics(self, cache_key: str) -> list[dict] | None:
        return self.get("financial_metrics", cache_key)

    def set_financial_metrics(self, cache_key: str, data: list[dict]) -> None:
        self.set("financial_metrics", cache_key, data)

    def get_line_items(self, cache_key: str) -> list[dict] | None:
        return self.get("line_items", cache_key)

    def set_line_items(self, cache_key: str, data: list[dict]) -> None:
        self.set("line_items", cache_key, data)

    def get_insider_trades(self, cache_key: str) -> list[dict] | None:
        return self.get("insider_trades", cache_key)

    def set_insider_trades(self, cache_key: str, data: list[dict]) -> None:
        self.set("insider_trades", cache_key, data)

    def get_company_news(self, cache_key: str) -> list[dict] | None:
        return self.get("company_news", cache_key)

    def set_company_news(self, cache_key: str, data: list[dict]) -> None:
        self.set("company_news", cache_key, data)


# Global cache instance (thread-safe lazy init)
_cache: Cache | None = None
_cache_lock = threading.Lock()


def get_cache() -> Cache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                _cache = Cache()
    return _cache
