"""Dynamic ticker universe — discovers interesting stocks to trade.

Uses Yahoo Finance (yfinance) to find top movers, most active, and momentum
stocks. Results are cached for 4 hours to avoid hammering the API.

Usage:
    universe = TickerUniverse()
    tickers = universe.scan()          # ~20-30 tickers
    trending = universe.get_trending()  # top 10 momentum stocks
"""

import threading
import time
from datetime import datetime, timedelta

from src.utils.logger import get_logger

log = get_logger(__name__)

# Always included in every scan
CORE_WATCHLIST = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"]

# Broader S&P 500 pool to screen from — top ~100 liquid large-caps by sector.
# We pull live data for these and rank by daily move / volume.
_SP500_SCREEN_POOL = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "META", "AVGO", "ORCL", "CRM",
    "AMD", "ADBE", "CSCO", "INTC", "QCOM", "INTU", "NOW", "IBM", "AMAT",
    "MU", "LRCX", "KLAC", "SNPS", "CDNS", "MRVL", "PANW", "CRWD", "FTNT",
    # Consumer / Retail
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "LOW",
    "CMG", "ABNB", "BKNG", "MAR", "LULU", "ROST", "DG",
    # Finance
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "C",
    "AXP", "SPGI", "ICE", "CB", "PGR", "MMC", "AON",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR",
    "BMY", "AMGN", "GILD", "ISRG", "MDT", "SYK", "BSX", "REGN", "VRTX",
    # Energy
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY",
    # Industrials
    "CAT", "GE", "HON", "UNP", "RTX", "BA", "LMT", "DE", "MMM", "UPS",
    # Communication
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
    # Misc large-cap
    "BRK-B", "PG", "KO", "PEP", "WMT", "PM", "MO", "CL", "EL",
]

# Cache duration
_CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours


class TickerUniverse:
    """Discovers and maintains a dynamic universe of tradeable tickers.

    Scans Yahoo Finance for top movers, most-active-by-volume, and momentum
    stocks, merging them with a core watchlist. Results are cached for 4 hours.
    """

    def __init__(self, core_watchlist: list[str] | None = None):
        self._core = [t.upper() for t in (core_watchlist or CORE_WATCHLIST)]
        self._lock = threading.Lock()

        # Scan cache
        self._cached_scan: list[str] = []
        self._cached_at: float = 0.0

        # Trending cache (separate TTL)
        self._cached_trending: list[str] = []
        self._trending_cached_at: float = 0.0

    # ── Public API ───────────────────────────────────────

    def scan(self) -> list[str]:
        """Return ~20-30 interesting tickers. Cached for 4 hours.

        Sources:
          1. Core watchlist (always included)
          2. S&P 500 top movers (biggest daily % gainers/losers)
          3. Most active by volume
          4. Current Alpaca holdings (if available)

        Returns the core watchlist on any error — never crashes.
        """
        with self._lock:
            if self._cached_scan and (time.monotonic() - self._cached_at) < _CACHE_TTL_SECONDS:
                log.debug("Returning cached scan (%d tickers)", len(self._cached_scan))
                return list(self._cached_scan)

        try:
            tickers = self._run_scan()
        except Exception as e:
            log.warning("Scan failed, returning core watchlist: %s", e)
            tickers = list(self._core)

        with self._lock:
            self._cached_scan = tickers
            self._cached_at = time.monotonic()

        return list(tickers)

    def get_trending(self) -> list[str]:
        """Return the top 10 momentum stocks (biggest positive % moves).

        Cached for 4 hours alongside the scan cache.
        """
        with self._lock:
            if self._cached_trending and (time.monotonic() - self._trending_cached_at) < _CACHE_TTL_SECONDS:
                return list(self._cached_trending)

        try:
            trending = self._find_momentum_stocks(top_n=10)
        except Exception as e:
            log.warning("Trending scan failed: %s", e)
            trending = list(self._core)[:10]

        with self._lock:
            self._cached_trending = trending
            self._trending_cached_at = time.monotonic()

        return list(trending)

    def invalidate_cache(self) -> None:
        """Force next scan() / get_trending() to re-fetch."""
        with self._lock:
            self._cached_at = 0.0
            self._trending_cached_at = 0.0

    # ── Internal ─────────────────────────────────────────

    def _run_scan(self) -> list[str]:
        """Execute the full scan pipeline. Returns deduplicated ticker list."""
        import yfinance as yf

        result: list[str] = list(self._core)  # always start with core

        # -- 1. Fetch daily data for the screening pool -----------------
        log.info("Scanning %d tickers for movers and volume...", len(_SP500_SCREEN_POOL))

        # Download 5 days of data for the pool (need at least 2 for pct_change)
        pool_tickers = [t for t in _SP500_SCREEN_POOL if t not in result]
        data = yf.download(
            pool_tickers,
            period="5d",
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if data.empty:
            log.warning("yfinance returned empty data for pool")
            return result

        # Build per-ticker stats
        stats: list[dict] = []
        for ticker in pool_tickers:
            try:
                if len(pool_tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker]

                if ticker_data.empty or len(ticker_data) < 2:
                    continue

                close = ticker_data["Close"].dropna()
                volume = ticker_data["Volume"].dropna()

                if len(close) < 2 or len(volume) < 1:
                    continue

                last_close = float(close.iloc[-1])
                prev_close = float(close.iloc[-2])
                if prev_close == 0:
                    continue
                pct_change = ((last_close - prev_close) / prev_close) * 100
                avg_volume = float(volume.tail(5).mean())
                last_volume = float(volume.iloc[-1])

                stats.append({
                    "ticker": ticker,
                    "pct_change": pct_change,
                    "abs_pct_change": abs(pct_change),
                    "avg_volume": avg_volume,
                    "last_volume": last_volume,
                    "last_close": last_close,
                })
            except Exception:
                # Skip individual ticker failures silently
                continue

        if not stats:
            log.warning("No stats computed — returning core watchlist")
            return result

        # -- 2. Top movers (biggest absolute % change) ------------------
        by_move = sorted(stats, key=lambda s: s["abs_pct_change"], reverse=True)
        movers = [s["ticker"] for s in by_move[:10]]
        log.info("Top movers: %s", movers)

        # -- 3. Most active by volume -----------------------------------
        by_volume = sorted(stats, key=lambda s: s["last_volume"], reverse=True)
        active = [s["ticker"] for s in by_volume[:8]]
        log.info("Most active: %s", active)

        # -- 4. Current Alpaca holdings ---------------------------------
        holdings = self._get_alpaca_holdings()
        if holdings:
            log.info("Current holdings: %s", holdings)

        # -- Merge all (deduplicated, preserving order) -----------------
        for ticker in movers + active + holdings:
            ticker = ticker.upper().strip()
            if ticker and ticker not in result:
                result.append(ticker)

        # Cap at 10 tickers — keeps analysis fast and avoids file descriptor exhaustion
        result = result[:10]

        log.info(
            "Scan complete: %d tickers (core=%d, movers=%d, active=%d, holdings=%d)",
            len(result), len(self._core), len(movers), len(active), len(holdings),
        )

        return result

    def _find_momentum_stocks(self, top_n: int = 10) -> list[str]:
        """Find top momentum stocks — biggest positive % moves."""
        import yfinance as yf

        data = yf.download(
            _SP500_SCREEN_POOL,
            period="5d",
            group_by="ticker",
            progress=False,
            threads=True,
        )

        if data.empty:
            return list(self._core)[:top_n]

        momentum: list[tuple[str, float]] = []
        for ticker in _SP500_SCREEN_POOL:
            try:
                ticker_data = data[ticker]
                close = ticker_data["Close"].dropna()
                if len(close) < 2:
                    continue
                prev = float(close.iloc[-2])
                if prev == 0:
                    continue
                pct = ((float(close.iloc[-1]) - prev) / prev) * 100
                if pct > 0:
                    momentum.append((ticker, pct))
            except Exception:
                continue

        momentum.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in momentum[:top_n]]

    @staticmethod
    def _get_alpaca_holdings() -> list[str]:
        """Get tickers of current Alpaca positions. Returns [] on failure."""
        try:
            from src.execution.alpaca import AlpacaBroker
            broker = AlpacaBroker()
            positions = broker.get_positions()
            return [p.ticker for p in positions if p.ticker]
        except Exception:
            # Alpaca not configured or unavailable — that's fine
            return []
