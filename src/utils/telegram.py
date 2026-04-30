"""Nova Trader Telegram alert sender."""

import os
import asyncio
import threading
from datetime import datetime

from src.utils.logger import get_logger

log = get_logger(__name__)

# Optional import — alerts degrade gracefully without the library
try:
    from telegram import Bot
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False


class TelegramAlerts:
    """Fire-and-forget trade alert sender."""

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        self._token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id and _HAS_TELEGRAM)

        if not _HAS_TELEGRAM:
            log.info("python-telegram-bot not installed — alerts disabled")
        elif not self._token or not self._chat_id:
            log.info("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — alerts disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _send(self, text: str) -> None:
        """Send a message. Non-blocking — fires in a background thread.

        Always sends as plain text (no parse_mode) to prevent formatting
        injection from user-controlled strings like reasoning or ticker names.
        """
        if not self._enabled:
            return

        def _do_send():
            try:
                bot = Bot(token=self._token)
                asyncio.run(
                    bot.send_message(
                        chat_id=self._chat_id,
                        text=text,
                    )
                )
            except Exception as e:
                log.error("Telegram send failed: %s", e)

        thread = threading.Thread(target=_do_send, daemon=True)
        thread.start()

    def send_trade(self, ticker: str, side: str, quantity: int,
                   price: float, status: str = "filled") -> None:
        """Alert on trade execution."""
        emoji = "🟢" if side == "buy" else "🔴"
        msg = (
            f"{emoji} Trade Executed\n"
            f"{ticker} — {side.upper()} {quantity} shares @ ${price:,.2f}\n"
            f"Status: {status}\n"
            f"{datetime.now().strftime('%H:%M:%S ET')}"
        )
        self._send(msg)


# ── Module-level singleton (thread-safe) ─────────────────

_alerts_instance: TelegramAlerts | None = None
_alerts_lock = threading.Lock()


def get_alerts() -> TelegramAlerts:
    """Get or create the singleton alert sender."""
    global _alerts_instance
    if _alerts_instance is None:
        with _alerts_lock:
            if _alerts_instance is None:
                _alerts_instance = TelegramAlerts()
    return _alerts_instance
