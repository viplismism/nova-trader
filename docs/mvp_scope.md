# Nova Trader MVP Scope

## Current Slice

Nova Trader is now scoped as a fast feedback MVP around a trade-signal agent council. The stable package runs ticker analysis through the existing engine, records the audit trail, produces clean signal cards, and supports follow-up questions grounded in the latest run. The CLI remains the primary developer/operator surface, while `nova web` adds a thin browser demo for investor and client walkthroughs.

The v1 council can be kept tight for demos: technical, fundamentals, news sentiment, web research, and SEC filings are enough to show the signal workflow with current-market and primary-source evidence. The full package can still run the default agent set when deeper coverage is useful. The output is a structured signal card per ticker: action, confidence, vote mix, risk limit, analyst reasoning, web/filing citations, decision risks, and what would change the call.

## In Scope For 3-4 Weeks

- Ticker analysis for public equities using the existing snapshot, analyst, consensus, risk, and portfolio pipeline.
- Clean terminal and browser presentation of agent activity, signal cards, final paragraphs, and grounded follow-up Q&A.
- Recorded run artifacts under the existing run directory: snapshots, views, signals, LLM calls, and recommendation JSON.
- Core demo mode with technical, fundamentals, sentiment, web research, and SEC filings agents; full council remains available.
- Live web research via Tavily when configured, with a best-effort DuckDuckGo fallback and explicit abstention when search is unavailable.
- SEC 10-K/10-Q retrieval with cached EDGAR fetches, passage search, and cited excerpts in the signal card.
- Optional adaptive research mode where MiniMax plans focused web/filing queries, Nova executes those searches, and MiniMax synthesizes a cited signal.
- Basic risk constraints: position cap, volatility-aware sizing, correlation multiplier, and portfolio-mode rules.
- Docker packaging for a small web deployment on AWS or a similar host.

## Out Of Scope For This MVP

- Custom quant runners, strategy backtesting UI, broker execution, order routing, or live trading.
- Deep portfolio integrations, custodian feeds, or proprietary data connectors.
- Complex RAG over internal documents beyond recorded Nova run artifacts and the first SEC filings slice.
- Multi-user auth, permissions, billing, or enterprise deployment controls.
- Options strategy selection beyond the current note that timeframe, borrow, IV/skew, and strike/expiry decide the short-vs-put instrument.

## Demo Surfaces

- `nova` opens the terminal workspace.
- `nova run --tickers AAPL,NVDA` runs the package directly.
- `nova web --port 8000` starts the browser demo.
- Browser follow-up questions answer from the latest recorded signal cards instead of free-form memory.

## Why This Was Hard

The hard part was not adding another UI; it was keeping the product coherent while multiple agents, deterministic metrics, explain-only LLM calls, risk sizing, portfolio rules, and run logs all happen at different points in the pipeline. Raw prompts and model responses are useful for audit, but they are bad product output. The work separates those layers: the engine still records everything, while the visible surfaces show clean reasoning, signal cards, and final paragraphs.

That separation matters for the MVP. It lets us show a credible workflow to Antipodes and Arpit without pretending the product already has heavy integrations or custom quant infrastructure. We can get feedback on the core loop first: ingest ticker data, live web results, and primary filing excerpts, run the council, show why each agent thought what it thought, apply basic risk constraints, and answer follow-up questions from the run record.
