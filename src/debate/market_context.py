"""One-paragraph live market context for the debate agents.

The debate's specialists/bear/synthesizer otherwise reason only from filings or
web search, which can lag the tape — a stale price anchor skews "what's already
priced in" judgments. This mirrors the reference pod's moomoo market snapshot
but sources everything from Nova's own data stack (src/tools/api), so no extra
broker dependency. Contract: NEVER raises — any failure returns '' and the
debate simply runs without the context block. NOVA_DEBATE_MARKET_CTX=0 is a
hard kill-switch (e.g. offline demos, deterministic tests).
"""

from __future__ import annotations

import datetime as _dt
import os

from src.tools import api as _api


def _fmt_cap(cap: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if cap >= div:
            return f"${cap / div:,.2f}{unit}"
    return f"${cap:,.0f}"


def build_market_context(ticker: str) -> str:
    """Return prose like:
    'Live market data for NVDA — last close $X (1-day change +Y%), 30-day range $A–$B, market cap $C.'
    Empty string on any failure or when NOVA_DEBATE_MARKET_CTX=0."""
    if os.environ.get("NOVA_DEBATE_MARKET_CTX", "").strip() == "0":
        return ""
    try:
        end = _dt.date.today()
        start = end - _dt.timedelta(days=60)  # ~40 trading days of dailies
        prices = _api.get_prices(ticker, start.isoformat(), end.isoformat())
        if not prices:
            return ""
        last = prices[-1]
        parts = [f"last close ${last.close:,.2f}"]
        if len(prices) >= 2:
            prev_close = prices[-2].close
            if prev_close:
                chg = (last.close - prev_close) / prev_close * 100
                parts[0] += f" (1-day change {chg:+.2f}%)"
        window = prices[-30:]
        lo = min(p.low for p in window)
        hi = max(p.high for p in window)
        parts.append(f"30-day range ${lo:,.2f}–${hi:,.2f}")
        try:
            cap = _api.get_market_cap(ticker, end.isoformat())
        except Exception:
            cap = None  # cap is a nice-to-have; keep the price context
        if cap:
            parts.append(f"market cap {_fmt_cap(cap)}")
        return f"Live market data for {ticker} — " + ", ".join(parts) + "."
    except Exception:
        return ""
