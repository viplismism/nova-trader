# Nova Trader

Minimal trading research, backtesting, and execution system.

The current runtime is a small core plus a CLI:

```text
CLI -> TradingService -> TradingPipeline -> signals -> risk -> portfolio decision -> optional execution
```

## Runtime Layout

```text
src/
  cli/
    app.py                  # argparse CLI with analyze/backtest commands
  core/
    config.py               # core configuration
    models.py               # portfolio, analysis, and backtest result models
    signals.py              # normalized signal payload helpers
    pipeline.py             # analysis pipeline
    service.py              # analysis, execution, and backtest facade
  data/                     # market data access, cache, memory, models
  alpha/features/           # factor, fundamentals, sentiment, technical scoring
  portfolio/                # deterministic vote-based portfolio construction
  risk/                     # position limit and volatility/correlation controls
  execution/                # execution bridge, broker adapters, audit, safety checks
  backtesting/              # portfolio simulation, costs, valuation, metrics
```

## CLI

Install dependencies with Poetry, then run either the script entrypoint or the module:

```bash
poetry install

poetry run nova analyze AAPL MSFT NVDA
poetry run nova analyze AAPL --months-back 6 --mode paper --execute

poetry run nova backtest AAPL MSFT --start-date 2024-01-01 --end-date 2024-12-31

poetry run python -m src.cli analyze AAPL
```

### Commands

- `analyze`: scores tickers, applies risk limits, builds deterministic actions, and can optionally execute them.
- `backtest`: reruns the same analysis flow on a business-day loop and reports equity curve and summary metrics.

### Key Flags

- `--start-date`, `--end-date`: explicit analysis or backtest window.
- `--months-back`: lookback window used when analysis start date is omitted, and per rebalance in backtests.
- `--cash`: starting cash / portfolio cash.
- `--mode dry_run|paper`: execution mode for analysis runs.
- `--execute`: submit decisions through the execution bridge.
- `--json`: print machine-readable output.

## Core Flow

1. `src/core/pipeline.py` loads price history from `src/data`.
2. Signal models in `src/alpha/features` score each ticker.
3. `src/risk` computes per-ticker headroom from volatility, correlation, exposure, cash, and drawdown inputs.
4. `src/portfolio/construction.py` converts usable signals into deterministic `buy`, `sell`, `short`, `cover`, or `hold` decisions.
5. `src/execution` can simulate orders or send paper orders through Alpaca.
6. `src/core/service.py` exposes the same logic to the CLI and to the built-in backtest loop.

## Notes

- The current CLI entrypoint is `src/cli/app.py`.
- The public core surface is exported from `src/core/__init__.py`.
- `nova`, `trade`, and `backtester` all point to the same CLI entrypoint in `pyproject.toml`.

## Docs

- [ARCHITECTURE.md](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/ARCHITECTURE.md): current runtime shape
- [TECH_SPEC.md](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/TECH_SPEC.md): module boundaries, flows, and current known gaps
- [GLOSSARY.md](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/GLOSSARY.md): repo terms and trading vocabulary used by the codebase
