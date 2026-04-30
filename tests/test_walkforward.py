"""Tests for walk-forward evaluation helpers.

These checks confirm that the evaluation utilities build the expected rolling
train/test windows and summarize metric dictionaries consistently across runs.
"""

from src.evals.walkforward import WalkForwardConfig, build_walkforward_windows, summarize_metrics


def test_build_walkforward_windows():
    windows = build_walkforward_windows(
        "2023-01-01",
        "2024-06-30",
        WalkForwardConfig(train_months=12, test_months=1, step_months=1),
    )

    assert windows
    assert windows[0].train_start == "2023-01-01"
    assert windows[0].train_end == "2023-12-31"
    assert windows[0].test_start == "2024-01-01"
    assert windows[0].test_end == "2024-01-31"


def test_summarize_metrics():
    summary = summarize_metrics(
        [
            {"sharpe_ratio": 1.0, "sortino_ratio": 1.5, "max_drawdown": -10.0},
            {"sharpe_ratio": 2.0, "sortino_ratio": 2.5, "max_drawdown": -8.0},
        ]
    )

    assert summary["runs"] == 2
    assert summary["avg_sharpe"] == 1.5
    assert summary["median_sharpe"] == 1.5
    assert summary["avg_sortino"] == 2.0
    assert summary["median_max_drawdown"] == -9.0
