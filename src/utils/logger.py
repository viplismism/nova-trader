"""Structured logging with correlation IDs for Nova Trader.

Every pipeline run gets a unique run_id. Every log entry includes:
  - timestamp, level, run_id, agent_id, ticker, message
  - Optional extra fields (latency_ms, data_source, etc.)

Usage:
    from src.utils.logger import get_logger, set_run_id

    log = get_logger(__name__)
    set_run_id("abc123")
    log.info("Fetching prices", agent_id="warren_buffett", ticker="AAPL")
"""

import json
import logging
import logging.handlers
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

# Thread-local storage for run_id
_context = threading.local()


def set_run_id(run_id: str | None = None) -> str:
    """Set the correlation ID for the current run. Returns the ID."""
    rid = run_id or uuid.uuid4().hex[:12]
    _context.run_id = rid
    return rid


def get_run_id() -> str:
    """Get the current run's correlation ID."""
    return getattr(_context, "run_id", "no-run")


class StructuredFormatter(logging.Formatter):
    """Emits JSON-structured log lines."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "run_id": get_run_id(),
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Add extra fields from the record
        for key in ("agent_id", "ticker", "latency_ms", "data_source",
                     "status", "error", "order_id", "action", "quantity",
                     "signal", "confidence", "mode"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])

        return json.dumps(entry, default=str)


class HumanFormatter(logging.Formatter):
    """Readable format for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        run_id = get_run_id()
        agent_id = getattr(record, "agent_id", "")
        ticker = getattr(record, "ticker", "")

        prefix_parts = [f"[{run_id[:8]}]"]
        if agent_id:
            prefix_parts.append(f"[{agent_id}]")
        if ticker:
            prefix_parts.append(f"[{ticker}]")

        prefix = " ".join(prefix_parts)
        msg = record.getMessage()

        return f"{color}{record.levelname:7s}{self.RESET} {prefix} {msg}"


class ExtraAdapter(logging.LoggerAdapter):
    """Logger adapter that accepts extra kwargs naturally.

    Usage:
        log.info("message", agent_id="foo", ticker="AAPL")
    """

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        # Move our custom fields into the extra dict
        extra = kwargs.get("extra", {})
        for key in list(kwargs.keys()):
            if key not in ("exc_info", "stack_info", "stacklevel", "extra"):
                extra[key] = kwargs.pop(key)
        kwargs["extra"] = extra
        return msg, kwargs


def _make_filter_factory():
    """Create a filter that adds default values for extra fields."""
    class DefaultsFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            for field in ("agent_id", "ticker", "latency_ms", "data_source",
                          "status", "error", "order_id", "action", "quantity",
                          "signal", "confidence", "mode"):
                if not hasattr(record, field):
                    setattr(record, field, None)
            return True
    return DefaultsFilter()


# Module-level setup (runs once, thread-safe)
_initialized = False
_init_lock = threading.Lock()
_json_handler: logging.Handler | None = None


def setup_logging(level: int = logging.INFO, json_output: bool = False) -> None:
    """Initialize the logging system. Call once at startup."""
    global _initialized, _json_handler
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        root = logging.getLogger()
        root.setLevel(level)

        # Remove existing handlers
        root.handlers.clear()

        # Console handler — human-readable
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(level)
        console.setFormatter(HumanFormatter())
        console.addFilter(_make_filter_factory())
        root.addHandler(console)

        if json_output:
            # JSON handler — structured output to file with rotation
            log_file = "nova-trader.log"
            _json_handler = logging.handlers.RotatingFileHandler(
                log_file, mode="a", maxBytes=10 * 1024 * 1024, backupCount=3,
            )
            _json_handler.setLevel(logging.DEBUG)
            _json_handler.setFormatter(StructuredFormatter())
            _json_handler.addFilter(_make_filter_factory())
            root.addHandler(_json_handler)
            try:
                os.chmod(log_file, 0o600)
            except OSError:
                pass

        # Quiet noisy libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("anthropic").setLevel(logging.WARNING)

        _initialized = True


def get_logger(name: str) -> ExtraAdapter:
    """Get a structured logger for a module.

    Usage:
        log = get_logger(__name__)
        log.info("Fetching data", agent_id="buffett", ticker="AAPL")
    """
    if not _initialized:
        setup_logging()
    return ExtraAdapter(logging.getLogger(name), {})
