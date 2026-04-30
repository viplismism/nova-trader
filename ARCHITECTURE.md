# Nova Trader Architecture

Nova Trader currently runs as a minimal Python core with a thin CLI.

## Main Path

```text
src/cli/app.py
  -> src/core/service.py
  -> src/core/pipeline.py
  -> src/alpha/features + src/data + src/risk + src/portfolio
  -> optional src/execution
  -> src/backtesting for simulation runs
```

## Core Modules

- `src/core/config.py`: core settings for cash, execution mode, signal model selection, weights, voting, and risk.
- `src/core/models.py`: shared state and result objects such as `PortfolioState`, `AnalysisResult`, and `BacktestResult`.
- `src/core/signals.py`: normalized signal payloads and health helpers.
- `src/core/pipeline.py`: pure analysis path for loading data, scoring signals, computing risk, and producing decisions.
- `src/core/service.py`: facade used by the CLI for `analyze` and `backtest`.

## Supporting Modules

- `src/data`: data access, caching, memory, and shared market data models.
- `src/alpha/features`: reusable factor, fundamentals, sentiment, and technical scorers.
- `src/portfolio`: deterministic vote-based action selection and sizing.
- `src/risk`: volatility, correlation, gross exposure, and drawdown-aware position limits.
- `src/execution`: execution bridge, broker abstraction, Alpaca paper broker, circuit breaker, and audit logging.
- `src/backtesting`: portfolio accounting, trading costs, valuation, metrics, and reusable backtest components.

## Analysis Flow

1. The CLI resolves command arguments and builds `CoreConfig`.
2. `TradingService.analyze()` initializes portfolio state and calls `TradingPipeline.analyze()`.
3. The pipeline fetches historical prices, runs configured signal models, computes risk headroom, and produces a deterministic decision per ticker.
4. If execution is enabled, `TradingService` passes decisions and a portfolio snapshot to `ExecutionBridge`.

## Backtest Flow

1. The CLI calls `TradingService.backtest()`.
2. The service loops over business days in the requested window.
3. For each day, it rebuilds portfolio state, runs the same analysis pipeline on the prior lookback window, and executes resulting actions through the backtesting trader.
4. The service returns an equity curve, trade count, final value, and summary metrics.

## Scope

- The current runtime does not depend on Hydra.
- The current runtime is not organized around multi-agent orchestration.
- The current user-facing entrypoint is the argparse CLI in `src/cli/app.py`.
- This document describes the code that exists now, not a future target layout.
