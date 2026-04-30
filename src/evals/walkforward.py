"""Walk-forward evaluation utilities.

This keeps the useful evaluation habit from Dexter without importing its
agent stack. The goal here is simple:
- define rolling train/test windows
- summarize repeated experiment runs consistently
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from typing import Any

from dateutil.relativedelta import relativedelta


@dataclass(slots=True)
class WalkForwardConfig:
    train_months: int = 12
    test_months: int = 1
    step_months: int = 1


@dataclass(slots=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def build_walkforward_windows(start_date: str, end_date: str, config: WalkForwardConfig | None = None) -> list[WalkForwardWindow]:
    cfg = config or WalkForwardConfig()
    overall_start = datetime.strptime(start_date, "%Y-%m-%d")
    overall_end = datetime.strptime(end_date, "%Y-%m-%d")

    windows: list[WalkForwardWindow] = []
    train_start = overall_start

    while True:
        train_end = train_start + relativedelta(months=cfg.train_months) - relativedelta(days=1)
        test_start = train_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=cfg.test_months) - relativedelta(days=1)

        if test_end > overall_end:
            break

        windows.append(
            WalkForwardWindow(
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_start=test_start.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
            )
        )
        train_start = train_start + relativedelta(months=cfg.step_months)

    return windows


def summarize_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize repeated backtest or evaluation runs."""
    if not runs:
        return {
            "runs": 0,
            "avg_sharpe": None,
            "median_sharpe": None,
            "avg_sortino": None,
            "median_max_drawdown": None,
        }

    sharpe = [r["sharpe_ratio"] for r in runs if r.get("sharpe_ratio") is not None]
    sortino = [r["sortino_ratio"] for r in runs if r.get("sortino_ratio") is not None]
    max_drawdown = [r["max_drawdown"] for r in runs if r.get("max_drawdown") is not None]

    return {
        "runs": len(runs),
        "avg_sharpe": mean(sharpe) if sharpe else None,
        "median_sharpe": median(sharpe) if sharpe else None,
        "avg_sortino": mean(sortino) if sortino else None,
        "median_max_drawdown": median(max_drawdown) if max_drawdown else None,
    }
