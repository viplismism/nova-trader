# Nova Trader

Nova Trader is a compact equity recommendation engine with optional long/short portfolio construction. It builds one market snapshot, gives each agent a typed view of that snapshot, aggregates signals, applies risk limits, and returns a structured recommendation with a mode-aware portfolio decision.

The current codebase is intentionally smaller than the earlier prototype: no LangChain, no graph runtime, no chat-message state machine. The product path is typed data in, typed recommendation out, with run artifacts saved for audit and replay.

## Current Runtime

```text
nova run
  -> build MarketSnapshot once
  -> slice typed agent views
  -> run registered analyst agents
  -> compute consensus
  -> run risk manager
  -> run portfolio manager
  -> write Recommendation + audit files

nova show <run_id>
  -> print a saved recommendation

nova rerun <run_id>
  -> replay against the saved snapshot
```

Every recorded run is written to:

```text
~/.nova-trader/runs/<run_id>/
```

Run artifacts include:

```text
metadata.json
snapshot.json
views.jsonl
signals.jsonl
llm.jsonl
recommendation.json
```

## Quick Start

```bash
poetry install
cp .env.example .env
```

Set at least one model key in `.env`. For the default OpenAI path:

```text
OPENAI_API_KEY=...
MODEL_PROVIDER=OpenAI
MODEL_NAME=gpt-4o-mini
```

MiniMax is also available through its OpenAI-compatible API:

```text
MINIMAX_API_KEY=...
MODEL_PROVIDER=MiniMax
MODEL_NAME=MiniMax-M2.7
```

Open the chat-style agent CLI:

```bash
poetry run nova
```

This opens a split-pane terminal workspace with a transcript, live run status,
and message box. Inside the chat, use plain commands:

```text
analyze AAPL,NVDA
mode long_short
model MiniMax MiniMax-M2.7
agents technical,valuation,warren_buffett
show last
rerun <run_id>
help
```

The explicit commands remain available for scripts and repeatable demos:

```bash
poetry run nova run --tickers AAPL,NVDA,TSLA
```

Run a smaller deterministic-agent demo plus the Buffett LLM persona:

```bash
poetry run nova run --tickers AAPL,NVDA --agents technical,fundamentals,growth,valuation,news_sentiment,insider_sentiment,warren_buffett
```

View or replay a saved run:

```bash
poetry run nova show <run_id>
poetry run nova rerun <run_id>
```

Backtester:

```bash
poetry run backtester --tickers AAPL --start-date 2024-01-01 --end-date 2024-12-31
```

## Agents

The registered analyst agents are defined in `src/registry.py`.

| Agent ID | Purpose |
|---|---|
| `technical` | Price trend, momentum, mean reversion, volatility |
| `fundamentals` | Profitability, growth, financial health, valuation ratios |
| `growth` | Revenue and earnings growth trends |
| `valuation` | DCF/comparables-style valuation signal |
| `news_sentiment` | News sentiment signal |
| `insider_sentiment` | Insider activity and sentiment signal |
| `warren_buffett` | Buffett-style quality/value persona using direct LLM JSON output |

Add a new agent by adding one `AgentSpec` in `src/registry.py` and implementing a runner with this shape:

```python
def run_agent(ctx: RunContext, view: SomeView, recorder=None) -> Signal:
    ...
```

The engine handles snapshot slicing, parallel execution, signal recording, aggregation, risk, portfolio construction, and final recommendation assembly.

## Architecture

```text
src/cli.py              CLI: run/show/rerun
src/engine.py           Runtime pipeline
src/snapshot.py         One-shot data fetch into MarketSnapshot
src/slicer.py           MarketSnapshot -> typed agent views
src/registry.py         Agent registry
src/aggregator.py       Signal -> Consensus
src/agents/             Analyst, risk, and portfolio agents
src/schemas/            Typed runtime contracts
src/runs.py             Persistent run recorder
src/utils/llm.py        Direct SDK/HTTP LLM adapter
src/backtesting/        Backtesting engine
```

## LLM Providers

Nova Trader uses direct SDK/HTTP adapters. The supported provider paths are:

- OpenAI
- Azure OpenAI
- OpenRouter
- DeepSeek
- Groq
- MiniMax
- xAI
- Ollama local models

No LangChain dependency is used in the application path.

## Product Direction

The product is a portfolio-aware equity recommendation engine with optional long/short construction. The default `research` mode shows the recommendation directly. In explicit `long_short` mode, a buy recommendation is not complete unless the portfolio manager can pair it with an acceptable short hedge, reduce it to available hedge capacity, or block it.

The language model is used for bounded JSON decisions and persona reasoning. Deterministic code owns snapshots, view boundaries, aggregation, risk limits, hedge pairing, persistence, and replay.
