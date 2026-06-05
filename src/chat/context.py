"""RunContext builders used by the chat interface."""

from __future__ import annotations

from datetime import date, datetime, timezone

from dateutil.relativedelta import relativedelta

from src.chat.models import ChatSettings
from src.runs import RunRecorder
from src.schemas.context import ModelConfig, RunContext, RunRequest
from src.schemas.portfolio import Portfolio, Position, RealizedGains
from src.schemas.snapshot import MarketSnapshot


def default_start_date() -> date:
    return (datetime.now() - relativedelta(months=3)).date()


def default_end_date() -> date:
    return datetime.now().date()


def build_context(tickers: list[str], settings: ChatSettings) -> RunContext:
    portfolio = Portfolio(
        cash=settings.initial_cash,
        margin_requirement=settings.margin_requirement,
        margin_used=0.0,
        positions={ticker: Position() for ticker in tickers},
        realized_gains={ticker: RealizedGains() for ticker in tickers},
    )
    request = RunRequest(
        tickers=tickers,
        start_date=default_start_date(),
        end_date=default_end_date(),
        portfolio=portfolio,
        model=ModelConfig(provider=settings.provider, name=settings.model),
        portfolio_mode=settings.portfolio_mode,
        show_reasoning=settings.show_reasoning,
        selected_agents=settings.agents,
    )
    return RunContext(request=request, as_of=datetime.now(timezone.utc))


def build_context_from_metadata(meta: dict) -> tuple[RunContext, MarketSnapshot]:
    request = RunRequest(
        tickers=meta["tickers"],
        start_date=date.fromisoformat(meta["start_date"]),
        end_date=date.fromisoformat(meta["end_date"]),
        portfolio=Portfolio(
            cash=100_000.0,
            margin_requirement=0.5,
            positions={ticker: Position() for ticker in meta["tickers"]},
            realized_gains={ticker: RealizedGains() for ticker in meta["tickers"]},
        ),
        model=ModelConfig(**meta["model"]),
        portfolio_mode=meta.get("portfolio_mode", "research"),
        selected_agents=meta.get("selected_agents", []),
    )
    ctx = RunContext(request=request, as_of=datetime.now(timezone.utc), seed=meta.get("seed"))
    snapshot = MarketSnapshot.model_validate(RunRecorder.load_snapshot_dict(meta["run_id"]))
    return ctx, snapshot
