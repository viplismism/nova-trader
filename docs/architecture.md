# Nova Trader Architecture

The architecture is split into smaller Swimlanes.io diagrams. Each diagram has a different POV and is intentionally compact so it renders cleanly.

## Diagrams

1. [System Architecture POV](./diagrams/01_product_pov.swimlanes)
   Compact product-level view of the system from user request to recommendation and optional execution.

2. [Recommendation Runtime](./diagrams/02_recommendation_runtime.swimlanes)
   Compact runtime view of how context, agents, risk, portfolio logic, and final answer fit together.

3. [Router And Workflow POV](./diagrams/03_model_router_pov.swimlanes)
   Compact router view showing how a BERT/SLM classifier selects product workflows.

## Planning

- [V0 Architecture](./v0_architecture.md)
  Layered architecture for the hedge-fund recommendation engine direction.

- [Three-Month Scope](./three_month_scope.md)
  Client-facing scope for shaping Nova Trader into an early hedge-fund recommendation system.

- [Product Flow](./product_flow.md)
  Plain-English flow of what Nova Trader serves to investment users.

## Current Product Shape

Nova Trader is a hedge-fund style recommendation system. It is not meant to be only a time-series forecasting tool. The product flow is:

1. User asks a market or portfolio question.
2. A fast router classifies intent and required context. Today this is a small model with deterministic fallback rules; a dedicated BERT/SLM router can replace it later.
3. The data pipeline prepares prices, fundamentals, news, sentiment, portfolio state, and history.
4. Analyst agents produce structured opinions.
5. Risk and portfolio layers adjust the recommendation for sizing, exposure, constraints, and mode-specific portfolio rules.
6. The recommendation engine returns an auditable answer with action, optional paired hedge, conviction, confidence, evidence, risks, and sizing.

## Implemented Now

- CLI and interactive terminal interface.
- Multi-agent analyst pipeline.
- Risk manager.
- Portfolio manager.
- Backtesting engine.
- Optional Alpaca paper execution bridge.
- Fast model-backed router with deterministic fallback.
- Typed recommendation schemas.
- Portfolio modes: `research`, `long_only`, and `long_short`. Only `long_short` enforces paired short hedges for opening buys.

## Not Implemented Yet

- Dedicated trained BERT/SLM router implementation.
- Dedicated evidence store.
- API server.
- Web dashboard.
- Recommendation ranking engine.
- Agent reliability scoring.
- Feedback and evaluation loop.
- Compliance/audit layer.
