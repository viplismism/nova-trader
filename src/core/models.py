"""Plain data models for the minimal trading core."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PositionState:
    long: int = 0
    short: int = 0


@dataclass(slots=True)
class PortfolioState:
    cash: float
    positions: dict[str, PositionState] = field(default_factory=dict)
    day_start_nav: float | None = None
    peak_nav: float | None = None

    def ensure_ticker(self, ticker: str) -> PositionState:
        if ticker not in self.positions:
            self.positions[ticker] = PositionState()
        return self.positions[ticker]

    def get_position_dict(self, ticker: str) -> dict[str, int]:
        position = self.ensure_ticker(ticker)
        return {"long": position.long, "short": position.short}

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": float(self.cash),
            "positions": {
                ticker: {"long": pos.long, "short": pos.short}
                for ticker, pos in self.positions.items()
            },
            "day_start_nav": self.day_start_nav,
            "peak_nav": self.peak_nav,
        }


@dataclass(slots=True)
class TickerReport:
    ticker: str
    current_price: float
    signals: dict[str, dict[str, Any]]
    risk: dict[str, Any]
    decision: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalysisResult:
    start_date: str
    end_date: str
    mode: str
    portfolio: PortfolioState
    reports: dict[str, TickerReport]
    execution: Any | None = None

    def decisions(self) -> dict[str, dict[str, Any]]:
        return {ticker: report.decision for ticker, report in self.reports.items()}

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "mode": self.mode,
            "portfolio": self.portfolio.to_dict(),
            "reports": {
                ticker: report.to_dict()
                for ticker, report in self.reports.items()
            },
        }
        if self.execution is not None:
            payload["execution"] = {
                "mode": self.execution.mode,
                "orders": [asdict(order) for order in self.execution.orders],
                "blocked_orders": self.execution.blocked_orders,
                "errors": self.execution.errors,
                "summary": self.execution.summary(),
            }
        return payload


@dataclass(slots=True)
class BacktestResult:
    start_date: str
    end_date: str
    metrics: dict[str, Any]
    equity_curve: list[dict[str, Any]]
    final_value: float
    trade_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
