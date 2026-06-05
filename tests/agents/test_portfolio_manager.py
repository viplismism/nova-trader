from datetime import date

from src.agents.portfolio import run_portfolio_manager
from src.schemas.context import RunContext, RunRequest
from src.schemas.portfolio import Portfolio, Position
from src.schemas.signals import Consensus, Limits, TickerLimit
from src.schemas.views import PortfolioView


def _ctx(tickers: list[str], portfolio_mode: str = "research") -> RunContext:
    return RunContext(
        request=RunRequest(
            tickers=tickers,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 1),
            portfolio=Portfolio(
                cash=100_000,
                margin_requirement=0.5,
                positions={ticker: Position() for ticker in tickers},
            ),
            portfolio_mode=portfolio_mode,
        )
    )


def _limit(ticker: str, price: float = 100.0) -> TickerLimit:
    return TickerLimit(
        ticker=ticker,
        current_price=price,
        max_position_dollars=10_000,
        max_shares=100,
        annualized_volatility=0.2,
        remaining_position_limit=10_000,
    )


def test_research_mode_allows_opening_buy_without_short_hedge():
    ctx = _ctx(["NVDA", "AAPL"])
    view = PortfolioView(
        portfolio=ctx.request.portfolio,
        consensus={
            "NVDA": Consensus(
                ticker="NVDA",
                direction="bullish",
                confidence=0.8,
                weighted_score=0.8,
                bull_count=2,
            ),
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
                weighted_score=0.0,
                neutral_count=2,
            ),
        },
    )
    limits = Limits(per_ticker={"NVDA": _limit("NVDA"), "AAPL": _limit("AAPL")})

    decisions = run_portfolio_manager(ctx, view, limits)

    assert decisions.hedge_plan.status == "not_required"
    assert decisions.hedge_plan.blocked_longs == []
    assert decisions.per_ticker["NVDA"].action == "buy"
    assert decisions.per_ticker["NVDA"].quantity > 0


def test_long_short_mode_blocks_opening_buy_when_no_short_hedge_exists():
    ctx = _ctx(["NVDA", "AAPL"], portfolio_mode="long_short")
    view = PortfolioView(
        portfolio=ctx.request.portfolio,
        consensus={
            "NVDA": Consensus(
                ticker="NVDA",
                direction="bullish",
                confidence=0.8,
                weighted_score=0.8,
                bull_count=2,
            ),
            "AAPL": Consensus(
                ticker="AAPL",
                direction="neutral",
                confidence=0.5,
                weighted_score=0.0,
                neutral_count=2,
            ),
        },
    )
    limits = Limits(per_ticker={"NVDA": _limit("NVDA"), "AAPL": _limit("AAPL")})

    decisions = run_portfolio_manager(ctx, view, limits)

    assert decisions.hedge_plan.status == "blocked"
    assert decisions.hedge_plan.blocked_longs == ["NVDA"]
    assert decisions.per_ticker["NVDA"].action == "hold"
    assert decisions.per_ticker["NVDA"].quantity == 0
    assert "no short hedge" in decisions.per_ticker["NVDA"].reasoning


def test_opening_buy_gets_hedge_pair_when_bearish_candidate_exists():
    ctx = _ctx(["NVDA", "AAPL"], portfolio_mode="long_short")
    view = PortfolioView(
        portfolio=ctx.request.portfolio,
        consensus={
            "NVDA": Consensus(
                ticker="NVDA",
                direction="bullish",
                confidence=0.8,
                weighted_score=0.8,
                bull_count=2,
            ),
            "AAPL": Consensus(
                ticker="AAPL",
                direction="bearish",
                confidence=0.7,
                weighted_score=-0.7,
                bear_count=2,
            ),
        },
    )
    limits = Limits(per_ticker={"NVDA": _limit("NVDA"), "AAPL": _limit("AAPL")})

    decisions = run_portfolio_manager(ctx, view, limits)

    assert decisions.hedge_plan.status in {"balanced", "partially_hedged"}
    assert decisions.hedge_plan.pairs
    assert decisions.per_ticker["NVDA"].action == "buy"
    assert decisions.per_ticker["NVDA"].hedge_pair_id == decisions.hedge_plan.pairs[0].pair_id
