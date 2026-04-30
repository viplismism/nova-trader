# Nova Trader Technical Specification

## Purpose

Nova Trader is a lean trading research, backtesting, and execution system built as a single Python package with a small command-line interface.

The current design goal is:

- keep the runtime small and debuggable
- make signal generation deterministic wherever possible
- separate data access, scoring, risk, portfolio decisions, and execution
- support both analysis and historical simulation from the same core path

## Non-Goals

The current repo is intentionally not:

- a microservices system
- a multi-agent orchestration platform
- a UI-first terminal product
- a high-frequency execution engine
- a distributed research platform

## Architecture Style

Nova Trader is a modular monolith.

That means:

- the code is split into focused modules
- the system runs as one application
- modules call each other directly in-process
- there is no service mesh, queue, or networked service boundary between core components

## Runtime Entry Points

The user-facing entrypoint is [src/cli/app.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/cli/app.py).

The CLI currently exposes two commands:

- `analyze`
- `backtest`

The public programmatic surface is exported from [src/core/__init__.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/__init__.py).

## Main Runtime Flow

The active runtime path is:

```text
CLI
  -> TradingService
  -> TradingPipeline
  -> data load
  -> signal scoring
  -> risk sizing
  -> portfolio decision
  -> optional execution
```

Main files:

- [src/cli/app.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/cli/app.py)
- [src/core/service.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/service.py)
- [src/core/pipeline.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/pipeline.py)
- [src/core/config.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/config.py)
- [src/core/models.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/models.py)

## Module Boundaries

### `src/cli`

Owns command parsing and display formatting.

Responsibilities:

- parse user inputs
- build `CoreConfig`
- call the service layer
- print either JSON or human-readable output

### `src/core`

Owns the application flow.

Responsibilities:

- configuration
- runtime state models
- analysis pipeline
- analyze/backtest service methods

### `src/data`

Owns market and company data access.

Responsibilities:

- fetch prices
- fetch financial metrics
- fetch company news
- fetch insider trades
- cache results
- normalize vendor payloads into internal schemas

### `src/alpha/features`

Owns deterministic scoring logic.

Responsibilities:

- factor scoring
- fundamental scoring
- sentiment scoring
- technical helper functions

### `src/portfolio`

Owns action selection.

Responsibilities:

- combine signals
- apply thresholds
- size actions
- return one deterministic action per ticker

### `src/risk`

Owns position limits and exposure math.

Responsibilities:

- volatility estimation
- correlation adjustment
- drawdown adjustment
- gross exposure control
- position headroom calculation

### `src/execution`

Owns trade submission and safety checks.

Responsibilities:

- map decisions to broker orders
- run circuit breaker checks
- record audit events
- support dry-run and paper trading modes

### `src/backtesting`

Owns historical simulation utilities.

Responsibilities:

- simulated portfolio accounting
- cost model
- trade execution in simulation
- valuation
- summary metrics

### `src/evals`

Owns higher-level experiment helpers such as walk-forward window generation.

## Analyze Flow

When the user runs `analyze`, the system behaves as follows:

1. The CLI resolves tickers, dates, cash, mode, and output style.
2. `TradingService.analyze()` creates or accepts a `PortfolioState`.
3. `TradingPipeline.analyze()` loads price history for requested tickers and any currently held tickers.
4. The pipeline runs the configured signal scorers.
5. The pipeline computes current portfolio statistics:
   - NAV
   - gross exposure
   - correlation matrix
   - drawdown multiplier
6. For each ticker, the pipeline computes a risk headroom estimate.
7. The portfolio module turns the signals plus risk cap into a final action:
   - `buy`
   - `sell`
   - `short`
   - `cover`
   - `hold`
8. The result is returned as an `AnalysisResult`.
9. If execution is enabled, the execution bridge applies safety checks and either simulates or places orders.

## Backtest Flow

When the user runs `backtest`, the system behaves as follows:

1. The CLI builds the service with the selected cash and lookback settings.
2. `TradingService.backtest()` creates a simulated portfolio.
3. The service loops over business days in the requested period.
4. For each day, it:
   - loads execution prices
   - converts the simulated portfolio into `PortfolioState`
   - reruns the same analysis logic on the prior lookback window
   - applies the resulting actions to the simulated portfolio
   - records value and exposure
5. The service computes summary metrics and returns a `BacktestResult`.

## Data Providers

The current data layer is centered around [src/data/api.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/data/api.py).

Current provider approach:

- primary structured provider: Financial Datasets endpoints
- fallback provider: Yahoo Finance
- local cache: SQLite-backed cache layer in [src/data/cache.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/data/cache.py)

The code currently pulls:

- prices
- financial metrics
- insider trades
- company news

## Default Signal Stack

The default enabled signal families are defined in [src/core/config.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/config.py):

- `factor`
- `fundamentals`
- `sentiment`

Default weights:

- factor: `1.20`
- fundamentals: `1.00`
- sentiment: `0.80`

This means the current architecture already prefers deterministic market and company data over softer signal sources.

## Decision Logic

Decision logic lives in [src/portfolio/construction.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/portfolio/construction.py).

The current decision system is:

- deterministic
- weighted by configured model weights
- filtered by minimum confidence
- gated by consensus thresholds
- limited by risk-derived share caps

This is intentionally simpler than an agent-debate architecture.

## Risk Logic

Risk logic lives in [src/risk/limits.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/risk/limits.py).

The current risk model considers:

- annualized volatility
- average correlation to held names
- gross exposure
- available cash
- current position value
- drawdown multiplier

Execution-time risk is stricter and separate:

- per-trade notional caps
- per-position caps
- gross exposure cap
- minimum confidence to trade
- daily loss and drawdown breakers
- kill switch

Those checks live in [src/execution/circuit_breaker.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/execution/circuit_breaker.py).

## Output Models

The main result models are:

- `PortfolioState`
- `TickerReport`
- `AnalysisResult`
- `BacktestResult`

They live in [src/core/models.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/models.py).

## Current Strengths

- clear modular boundaries
- minimal CLI
- deterministic action selection
- shared core logic for analyze and backtest
- explicit risk layer
- execution safety checks
- local tests for the core path

## Current Known Gaps

### Data normalization

Some upstream financial metric values still need normalization. This is visible in live analysis output where certain valuation fields appear mis-scaled.

### Backtest speed

The current backtest loop re-runs analysis and data access per business day, which makes longer backtests slower than they should be.

### Legacy overlap

The repo is much cleaner than before, but some older support modules still exist alongside the new core layout.

### Output polish

The CLI is functional and compact, but the human-readable output can still be improved.

## What To Keep Stable

These boundaries should remain stable unless there is a strong reason to change them:

- CLI only handles user input and printing
- core owns orchestration
- data owns external data access
- alpha owns scoring
- portfolio owns decisions
- risk owns limits
- execution owns broker-facing actions and hard safety

## Immediate Design Direction

The next steps should improve correctness and speed without changing the basic architecture:

1. fix financial metric normalization in the data layer
2. speed up the backtest loop by reducing repeated data fetches
3. improve CLI output formatting
4. continue adding deterministic signal logic before adding more orchestration layers
