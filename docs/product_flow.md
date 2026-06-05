# Nova Trader Product Flow

Nova Trader is moving toward a portfolio-aware equity recommendation engine with optional long/short construction. The product should not answer only "buy this stock." It should answer "what is the signal, how should the position be sized, what evidence supports it, and, if we are in long/short mode, what should we short against it?"

## What We Serve

For an investment user, the served output is a structured recommendation package:

- **Question understanding**: The system identifies whether the user is asking for a single-stock view, portfolio review, risk exposure, valuation, news sentiment, comparison, or backtest.
- **Research evidence**: The system gathers prices, fundamentals, valuation inputs, news, sentiment, insider activity, and portfolio context.
- **Agent opinions**: Analyst agents produce structured bullish, bearish, or neutral opinions with confidence and reasoning.
- **Risk constraints**: The risk layer calculates volatility, current exposure, available position limits, and correlation adjustments.
- **Portfolio decision**: The portfolio manager turns research signals into decisions. In `research` mode it shows the recommendation directly. In `long_short` mode, every opening buy must have a corresponding short hedge or the buy is reduced/blocked.
- **Recommendation answer**: The user sees the action, optional paired short candidate, sizing, conviction, confidence, evidence, risks, and what would change the recommendation.
- **Optional backtest/paper execution**: The same decision shape can be simulated historically or sent to paper execution later, but V0 should remain research-first.

## Runtime Flow

1. **User asks a question**
   Example: "Should we buy NVDA over the next quarter?"

2. **Router creates a structured route**
   The query is mapped to an intent, tickers, horizon, required modules, and whether portfolio context is needed.

3. **Workflow selector chooses the run**
   A single-stock recommendation might run fundamentals, valuation, technicals, news sentiment, risk, and portfolio management.

4. **Data layer gathers evidence**
   The system pulls market prices, financial metrics, line items, news, sentiment, insider trades, and current portfolio state.

5. **Analyst agents score the asset**
   Agents produce structured opinions. One agent may like valuation, another may dislike momentum, and another may flag news risk.

6. **Risk manager applies portfolio constraints**
   The risk layer calculates allowed position sizes based on cash, margin, volatility, and correlation.

7. **Portfolio manager builds the decision**
   The portfolio layer chooses the action and position size. In `long_short` mode it also finds the weakest or most suitable short candidate. If no short hedge is available in that mode, the opening buy is not allowed as a complete recommendation.

8. **Structured recommendation is returned**
   The answer is not just prose. It should include action, optional paired hedge, size, confidence, conviction, evidence, risks, and what would change the view.

## Product Principle

The system computes the recommendation. The language model explains it.

That means deterministic code should own routing contracts, evidence shape, risk limits, hedge pairing, sizing, and auditability. LLMs can help summarize, classify, reason over evidence, and produce readable explanations, but they should not be the only source of the final trading decision.

## First-Cut Boundary

For the first cut, the product should serve decision support, not autonomous capital deployment. The user gets a recommendation package that can be reviewed, challenged, logged, and later backtested. Long/short hedge enforcement is available as an explicit mode, not a hidden default. Live execution should stay out of scope until the recommendation engine is reliable enough to audit.
