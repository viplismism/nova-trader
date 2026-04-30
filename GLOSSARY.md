# Nova Trader Glossary

This glossary explains the terms used in the current codebase.

For the full system shape, read [TECH_SPEC.md](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/TECH_SPEC.md).

## System Terms

### CLI
The command-line interface. In this repo, the CLI entrypoint is [src/cli/app.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/cli/app.py). It exposes the `analyze` and `backtest` commands.

### Core
The small orchestration layer that wires the whole runtime together. In this repo, `core` means [src/core/config.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/config.py), [src/core/models.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/models.py), [src/core/pipeline.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/pipeline.py), and [src/core/service.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/service.py).

### Pipeline
A fixed sequence of steps that turns inputs into outputs. In Nova Trader, the pipeline is:

`data -> signals -> risk -> decision -> optional execution`

### Service
A thin layer that exposes the main use cases. `TradingService` exposes `analyze()` and `backtest()` and hides the lower-level wiring.

### Modular Monolith
A single application with separate internal modules. Nova Trader is a modular monolith, not a microservices system. The boundaries are Python modules, not separately deployed network services.

### Microservices
An architecture where different parts of the system run as separate services and talk over the network. That is not how this repo works today.

### Adapter
A module that connects Nova Trader to an outside system or a different interface. Examples are the data adapter in [src/data/api.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/data/api.py) and the broker adapter in [src/execution/alpaca.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/execution/alpaca.py).

### Contract
A stable data shape passed between modules. In this repo, contracts are plain dataclasses, typed dictionaries, and Pydantic models.

## Analysis Terms

### Signal
A structured view on a ticker, usually `bullish`, `bearish`, or `neutral`, plus confidence and reasoning.

### Scorer
A function that turns raw data into a signal. Examples live in [src/alpha/features](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/alpha/features).

### Factor
A systematic investing lens like `value`, `momentum`, or `quality`. In this repo, the factor scorer also blends `low volatility`, `defensive`, and `multi-factor`.

### Fundamentals
Signals derived from company financial health, such as profitability, valuation, growth, and leverage.

### Sentiment
Signals derived from softer information like news or insider trading activity.

### Confidence
How strong a signal is on a 0 to 100 scale. Confidence is used when deciding whether a signal is strong enough to count and how much weight it should receive.

### Voting
The deterministic step that combines multiple signals into one final action. Voting logic lives in [src/portfolio/construction.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/portfolio/construction.py).

### Weighted Score
The combined directional score produced by the voting logic after applying signal confidence and configured model weights.

### Consensus
How many usable signals agree on direction. The voting logic uses both score strength and consensus percentage.

## Portfolio Terms

### Portfolio State
The live analysis view of the account: cash, current positions, and optional NAV markers. This is modeled by `PortfolioState` in [src/core/models.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/core/models.py).

### Position
A holding in one ticker. In this repo, a position tracks `long` shares and `short` shares separately.

### Long
Owning shares and benefiting when the price rises.

### Short
Borrowing and selling shares first, then buying them back later. A short position benefits when the price falls.

### Buy
Open or add to a long position.

### Sell
Reduce or close a long position.

### Short Action
Open or add to a short position.

### Cover
Reduce or close a short position.

### Hold
Take no action.

### NAV
Net Asset Value. A snapshot of total portfolio value after adding cash and longs and subtracting shorts.

### Gross Exposure
Total absolute market exposure, ignoring direction. Long and short exposures are both added.

### Net Exposure
Directional exposure after subtracting shorts from longs.

### Position Limit
The maximum size allowed for a ticker, usually measured in dollars or as a percentage of portfolio value.

## Risk Terms

### Risk Headroom
The remaining room available for a position after applying the configured risk limits.

### Volatility
How much a price series moves around. In the repo, volatility is estimated from close-to-close returns.

### Annualized Volatility
Daily volatility scaled to a yearly basis.

### Correlation
How closely two price series move together. High correlation means two names may add less diversification than they appear to.

### Drawdown
The drop from a prior peak NAV to the current NAV.

### Drawdown Multiplier
A position-size reduction factor applied after the portfolio has fallen beyond a configured drawdown threshold.

### Circuit Breaker
Hard execution-time safety checks that can block trades even if the analysis logic wants to place them. The checks live in [src/execution/circuit_breaker.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/execution/circuit_breaker.py).

### Kill Switch
A manual hard stop that halts execution regardless of signals. This is implemented as a file-based sentinel in the circuit breaker module.

## Data Terms

### Data Provider
An external source for prices, fundamentals, news, or other market data. The current data layer uses Financial Datasets first and Yahoo Finance as fallback in [src/data/api.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/data/api.py).

### Fallback
A secondary source used when the primary source fails or returns nothing.

### Cache
A local store of recently fetched data to avoid repeating the same external request. Cache logic lives in [src/data/cache.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/data/cache.py).

### Schema
The expected shape of data. This repo uses Pydantic models in [src/data/models.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/data/models.py) to define those schemas.

### Normalization
Converting external data into one internal shape so the rest of the code does not depend on vendor-specific details.

## Execution Terms

### Execution Bridge
The layer that translates decisions into broker actions after safety checks. It lives in [src/execution/bridge.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/execution/bridge.py).

### Dry Run
A mode where orders are simulated and logged but not sent to a broker.

### Paper Trading
A mode where orders are sent to a simulated brokerage account rather than a live brokerage account.

### Audit Log
A persistent record of what the system tried to do and what actually happened during execution.

## Backtesting Terms

### Backtest
A historical simulation of the trading logic over a chosen date range.

### Equity Curve
The time series of portfolio value produced by a backtest.

### Slippage
The difference between the reference price and the price you assume you actually receive in execution.

### Commission
Per-trade or per-share execution cost assumed in simulation.

### Sharpe Ratio
A simple risk-adjusted return metric based on average excess return divided by return volatility.

### Sortino Ratio
A variation of Sharpe that focuses on downside volatility instead of total volatility.

### Max Drawdown
The worst observed peak-to-trough decline in portfolio value during a backtest.

### Walk-Forward Evaluation
A repeated train/test process over rolling date windows. Helpers for this live in [src/evals/walkforward.py](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/src/evals/walkforward.py).

## Repo-Specific Terms

### `analyze`
The main command that runs data loading, signal scoring, risk sizing, and deterministic decision generation for one or more tickers.

### `backtest`
The command that reruns the analysis flow over a historical date range and simulates trading outcomes.

### `factor`, `fundamentals`, `sentiment`
The three signal families currently enabled by default in `CoreConfig`.

### Tech Spec
A design document that explains the runtime, module boundaries, flows, assumptions, and known gaps. In this repo, that document is [TECH_SPEC.md](/Users/vipul.m/Nova/Projects/active/ai-traders/nova-trader/TECH_SPEC.md).
