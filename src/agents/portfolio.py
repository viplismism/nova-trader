"""Portfolio manager on the new contract.

Input:  RunContext, PortfolioView (portfolio + consensus), Limits.
Output: Decisions (per-ticker action + quantity + hedge plan).

For Round 1 we use a deterministic decision rule rather than the LLM:
  - bullish + room in limits + cash available -> buy up to max_shares (sized by confidence)
  - bearish + existing long -> sell long
  - bearish + room to short -> short
  - everything else -> hold

This makes the test runnable without LLM cost and gives us a clear
baseline. We can swap in the LLM-driven decision later — the contract
doesn't change.
"""

from __future__ import annotations

import logging
import uuid

from src.schemas.context import RunContext
from src.schemas.signals import (
    Consensus,
    Decisions,
    HedgePair,
    HedgePlan,
    Limits,
    TickerDecision,
)
from src.schemas.views import PortfolioView
from src.utils.progress import progress

logger = logging.getLogger(__name__)

AGENT_ID = "portfolio_manager"


def _decide_one(
    ticker: str,
    consensus: Consensus | None,
    limit,
    has_long: bool,
    has_short: bool,
) -> TickerDecision:
    """Decision rule for a single ticker."""
    if consensus is None or limit is None or limit.current_price <= 0:
        return TickerDecision(ticker=ticker, action="hold", reasoning="No consensus or price data")

    if consensus.direction == "bullish":
        if limit.max_shares <= 0:
            return TickerDecision(
                ticker=ticker, action="hold",
                confidence=consensus.confidence,
                reasoning=f"Bullish but no room (max_shares={limit.max_shares})",
            )
        # Size by confidence: spend confidence * max_position
        target_dollars = limit.max_position_dollars * consensus.confidence
        qty = int(target_dollars // limit.current_price)
        if qty <= 0:
            return TickerDecision(
                ticker=ticker, action="hold",
                confidence=consensus.confidence,
                reasoning="Bullish but confidence-sized quantity rounds to 0",
            )
        return TickerDecision(
            ticker=ticker, action="buy", quantity=qty,
            confidence=consensus.confidence,
            reasoning=f"Bullish {consensus.confidence:.0%}, "
                      f"{consensus.bull_count} bulls vs {consensus.bear_count} bears",
        )

    if consensus.direction == "bearish":
        if has_long:
            return TickerDecision(
                ticker=ticker, action="sell", quantity=0,  # quantity TBD by caller
                confidence=consensus.confidence,
                reasoning="Bearish on existing long position",
            )
        # Could initiate short, but in Round 1 we keep this simple
        return TickerDecision(
            ticker=ticker, action="hold",
            confidence=consensus.confidence,
            reasoning=f"Bearish {consensus.confidence:.0%} but no position to exit",
        )

    return TickerDecision(
        ticker=ticker, action="hold",
        confidence=consensus.confidence,
        reasoning=f"Neutral consensus "
                  f"({consensus.bull_count} bull / {consensus.bear_count} bear)",
    )


def run_portfolio_manager(ctx: RunContext, view: PortfolioView, limits: Limits) -> Decisions:
    decisions = Decisions()
    portfolio_mode = getattr(ctx.request, "portfolio_mode", "research")

    for ticker in ctx.tickers:
        progress.update_status(AGENT_ID, ticker, "Deciding")
        consensus = view.consensus.get(ticker)
        limit = limits.per_ticker.get(ticker)
        pos = view.portfolio.positions.get(ticker)
        decisions.per_ticker[ticker] = _decide_one(
            ticker=ticker,
            consensus=consensus,
            limit=limit,
            has_long=bool(pos and pos.long > 0),
            has_short=bool(pos and pos.short > 0),
        )

    # Simple hedge plan: pair the strongest bullish with the strongest bearish.
    bull = [t for t, d in decisions.per_ticker.items() if d.action == "buy"]
    bear = [t for t in ctx.tickers
            if (c := view.consensus.get(t)) and c.direction == "bearish"]

    if portfolio_mode != "long_short":
        progress.update_status(AGENT_ID, None, "Done")
        return decisions

    if bull and bear:
        # Pair the most confident bull with the most confident bear.
        bull_sorted = sorted(bull, key=lambda t: -decisions.per_ticker[t].confidence)
        bear_sorted = sorted(bear, key=lambda t: -view.consensus[t].confidence)
        long_t = bull_sorted[0]
        short_t = bear_sorted[0]
        long_dec = decisions.per_ticker[long_t]
        long_limit = limits.per_ticker.get(long_t)
        short_limit = limits.per_ticker.get(short_t)
        if long_limit and short_limit and long_limit.current_price > 0 and short_limit.current_price > 0:
            long_notional = long_dec.quantity * long_limit.current_price
            short_qty = int(long_notional // short_limit.current_price)
            short_notional = short_qty * short_limit.current_price
            pair_id = f"pair_{uuid.uuid4().hex[:6]}"
            pair = HedgePair(
                pair_id=pair_id,
                long_ticker=long_t,
                short_ticker=short_t,
                long_quantity=long_dec.quantity,
                short_quantity=short_qty,
                long_notional=long_notional,
                short_notional=short_notional,
                hedge_ratio=(short_notional / long_notional) if long_notional > 0 else 0.0,
            )
            decisions.per_ticker[long_t].hedge_pair_id = pair_id
            decisions.hedge_plan = HedgePlan(
                strategy="equity_long_short",
                target_hedge_ratio=1.0,
                status="balanced" if pair.hedge_ratio >= 0.9 else "partially_hedged",
                pairs=[pair],
                long_notional=long_notional,
                short_notional=short_notional,
                net_notional=long_notional - short_notional,
            )
    elif bull:
        # Bull without a hedge candidate — block the trade itself. In equity
        # long/short mode a standalone opening buy is not a complete decision.
        for ticker in bull:
            old = decisions.per_ticker[ticker]
            decisions.per_ticker[ticker] = TickerDecision(
                ticker=ticker,
                action="hold",
                quantity=0,
                confidence=old.confidence,
                reasoning="Blocked: no short hedge candidate available",
            )
        decisions.hedge_plan = HedgePlan(
            status="blocked",
            blocked_longs=bull,
        )

    progress.update_status(AGENT_ID, None, "Done")
    return decisions
