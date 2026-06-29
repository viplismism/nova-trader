<aside>
🧭

Nova Trader is an investment-research system built from two complementary engines:

- **Signals engine:** eight deterministic agents plus two LLM-assisted research agents that convert market, financial, news, insider, web, and SEC data into directional signals.
- **Debate engine:** an LLM research pod that plans, investigates, challenges, and synthesizes an investment memo from evidence.

The goal of this page is to make every agent easy to understand at two levels: **what it is meant to contribute** and **what it actually does technically**.

</aside>

## System architecture

Nova has three major layers:

1. **Evidence collection** - market prices, fundamentals, filings, news, insider trades, and live web sources.
2. **Agent interpretation** - deterministic agents and LLM research agents turn raw evidence into views, scores, citations, and risks.
3. **Decision layer** - consensus, risk, and portfolio logic translate the agent views into ratings, position limits, actions, and hedge plans.

### Engine overview

- **Signals engine:** eight deterministic rules-based agents plus two LLM-assisted agents: Adaptive Research and Warren Buffett. The rules-based agents are repeatable; the LLM-assisted agents add planning, synthesis, or refinement.
- **Decision layer:** combines signals, applies risk constraints, and generates portfolio actions.
- **Debate engine:** a research pod made of a planner, specialists, bear challenger, and synthesizer. This layer is evidence-led and memo-oriented rather than purely score-oriented.

---

## Part 1 — Signals engine

The Signals engine is the quantitative and data-driven core of Nova. Most agents are deterministic and rules-based; Adaptive Research and Warren Buffett are LLM-assisted. Each agent looks at one lens of the company, emits a bullish / bearish / neutral view, and attaches confidence plus supporting factors.

## 1. Technical Analyst · `technical`

### What this agent is meant to do

The Technical Analyst is Nova's market-behavior reader. Its job is to answer: **what is the price action saying right now?**

It does not try to decide whether the company is fundamentally good or bad. Instead, it studies historical price patterns to detect trend, momentum, mean reversion, volatility setup, and statistical behavior. This makes it useful for timing, entry quality, short-term risk, and confirming or challenging the longer-term thesis.

### What it actually does

- **Input data:** historical daily prices.
- **Core method:** a five-strategy ensemble:
    - Trend: EMA 8 / 21 / 55 plus ADX.
    - Mean reversion: Z-score, Bollinger Bands, and RSI.
    - Momentum.
    - Volatility.
    - Statistical arbitrage.
- **Weights:**
    - Trend: 25%.
    - Momentum: 25%.
    - Mean reversion: 20%.
    - Volatility: 15%.
    - Statistical arbitrage: 15%.
- **Output:** bullish / bearish / neutral signal, confidence score, and each strategy's sub-signal as a key factor.

---

## 2. Fundamentals Analyst · `fundamentals`

### What this agent is meant to do

The Fundamentals Analyst is Nova's business-quality checker. Its job is to answer: **does this company look financially strong based on its latest reported metrics?**

It focuses on profitability, growth, balance-sheet health, and basic valuation sanity. This agent is useful for quickly separating companies with strong operating metrics from companies where the business quality or financial condition looks weak.

### What it actually does

- **Input data:** most recent financial metrics.
- **Core method:** four sub-scores, each made from three checks:
    - **Profitability:** ROE > 15%, net margin > 20%, operating margin > 15%.
    - **Growth:** revenue growth, earnings growth, and book-value growth > 10%.
    - **Health:** current ratio > 1.5, debt/equity < 0.5, free cash flow per share > 80% of EPS.
    - **Valuation:** P/E < 25, P/B < 3, P/S < 5.
- **Decision rule:**
    - Bullish if at least three sub-scores are bullish.
    - Bearish if at least three sub-scores are bearish.
    - Otherwise neutral.
- **Output:** directional signal and confidence scaled by the number of confirming sub-scores.

---

## 3. Growth Analyst · `growth`

### What this agent is meant to do

The Growth Analyst is Nova's compounding-potential lens. Its job is to answer: **is this company growing in a way that looks durable and financially healthy?**

It looks beyond one-period numbers and checks whether growth trends, valuation, margins, and balance-sheet health support a growth thesis. This is especially useful for companies where the investment case depends on expansion, reinvestment, and future earnings power.

### What it actually does

- **Input data:** at least four periods of metrics and line items.
- **Core method:** weighted blend of:
    - Growth trends: 45%.
    - Valuation: 25%.
    - Margin trends: 20%.
    - Financial health: 10%.
- **Decision rule:**
    - Bullish if weighted score > 0.60.
    - Bearish if weighted score < 0.40.
    - Neutral otherwise.
- **Output:** bullish / bearish / neutral growth signal.

---

## 4. Valuation Analyst · `valuation`

### What this agent is meant to do

The Valuation Analyst is Nova's intrinsic-value engine. Its job is to answer: **what is the company worth compared with what the market is currently pricing?**

It does not rely on a single valuation technique. Instead, it blends several models so that the conclusion is not overly dependent on one assumption set. It also produces a 12-month price target, which makes the valuation output easier to connect to portfolio decisions and upside/downside framing.

### What it actually does

- **Input data:** free cash flow history, financial metrics, market cap, and shares outstanding.
- **Core method:** four intrinsic-value methods blended by weight:
    - DCF: 35%.
    - Owner Earnings: 35%.
    - EV/EBITDA: 20%.
    - Residual Income: 10%.
- **DCF detail:** multi-stage model with bear / base / bull scenarios and CAPM-based WACC.
- **Gap calculation:** compares weighted intrinsic value against current market cap.
- **12-month target logic:**
    - Fair value per share = current price × (1 + valuation gap).
    - Target price = fair value × (1 + cost of equity; currently ~10.5%, from CAPM: 4.5% risk-free + 6% equity premium).
    - Reports target price, fair value, and upside percentage.
- **Decision rule:**
    - Bullish if gap > +15%.
    - Bearish if gap < -15%.
    - Neutral otherwise.
- **Output:** direction, confidence, valuation gap, fair value, target price, and upside.

---

## 5. News Sentiment Analyst · `news_sentiment`

### What this agent is meant to do

The News Sentiment Analyst is Nova's news-flow reader. Its job is to answer: **is recent company news helping or hurting the investment narrative?**

It captures the market narrative around the company: product updates, guidance, management commentary, legal issues, analyst reactions, customer momentum, competitive pressure, and other headline-level catalysts.

### What it actually does

- **Input data:** recent company news headlines, including vendor sentiment tags when available.
- **Core method:**
    - Uses vendor sentiment label if present.
    - Falls back to a keyword classifier if vendor sentiment is missing.
    - Aggregates positive versus negative headline signals.
- **Score:** net positive-minus-negative score across headlines.
- **Decision rule:**
    - Bullish if net > +0.10.
    - Bearish if net < -0.10.
    - Neutral otherwise.
- **Output:** sentiment direction and confidence.

---

## 6. Insider Sentiment Analyst · `insider_sentiment`

### What this agent is meant to do

The Insider Sentiment Analyst is Nova's management-and-insider behavior lens. Its job is to answer: **are insiders buying or selling in a way that changes the thesis?**

Insider buying can indicate confidence, while heavy insider selling can introduce caution. This signal should not be treated as a full investment thesis by itself, but it is a useful behavioural input.

### What it actually does

- **Input data:** insider transactions.
- **Core method:** compares insider buys versus insider sells.
- **Score:** `(buys - sells) / total transactions`.
- **Decision rule:**
    - Bullish if net > +0.10.
    - Bearish if net < -0.10.
    - Neutral otherwise.
- **Output:** direction and reasoning with buy/sell counts.

---

## 7. Web Research Analyst · `web_research`

### What this agent is meant to do

The Web Research Analyst is Nova's external-research scanner. Its job is to answer: **what does the live web say about the company right now?**

This agent helps catch information that may not be present in structured financial data yet: product announcements, legal/regulatory updates, competitive news, demand commentary, sector narratives, and recent market discussion.

### What it actually does

- **Input data:** live web search results.
- **Search provider:** Tavily if a key is configured; otherwise DuckDuckGo.
- **Core method:** counts constructive versus risk keywords across titles and snippets.
- **Score:** `(constructive keywords - risk keywords) / total keywords`.
- **Decision rule:**
    - Bullish if score ≥ +0.15.
    - Bearish if score ≤ -0.15.
    - Neutral otherwise.
- **Output:** direction, confidence, and up to eight web-source citation chips.

---

## 8. SEC Filings Analyst · `sec_filings`

### What this agent is meant to do

The SEC Filings Analyst is Nova's official-disclosure reader. Its job is to answer: **what do the company's own filings reveal about the business, risks, and management commentary?**

This is the highest-trust source layer because it uses primary documents. It helps ground the thesis in audited or formally filed company disclosures instead of relying only on headlines or market narratives.

### What it actually does

- **Input data:** real SEC 10-K and 10-Q filings fetched from EDGAR.
- **Processing:** filings are chunked, tagged by Item section, and retrieved using BM25.
- **Core method:** counts positive versus negative terms across retrieved excerpts.
- **Score:** `(positive terms - negative terms) / total terms`.
- **Decision rule:**
    - Bullish if score ≥ +0.15.
    - Bearish if score ≤ -0.15.
    - Neutral otherwise.
- **Output:** direction, confidence, and up to eight filing-citation chips linking back to EDGAR documents.

---

## 9. Adaptive Research Analyst · `adaptive_research`

### What this agent is meant to do

The Adaptive Research Analyst is Nova's flexible investigator. Its job is to answer: **what should we research for this ticker, and what does that research imply?**

Unlike a fixed rules-only agent, it can plan a custom research path. It decides what to look for across metrics, news, filings, and web results, then synthesizes the evidence into a signal. This is useful when a ticker has a non-standard story or when the most important question changes by company.

### What it actually does

- **Input data:** metrics, news, filings, and web results fetched on demand.
- **Core workflow:**
    1. An LLM plans the research focus and search queries.
    2. Nova retrieves the requested web and filing evidence.
    3. The LLM synthesizes a memo-like signal with confidence and key findings.
- **Fallback:** neutral if the LLM is unavailable.
- **Output:** direction, confidence, key findings, web citations, and filing citations.

---

## 10. Warren Buffett Analyst · `warren_buffett`

### What this agent is meant to do

The Warren Buffett Analyst is Nova's quality-at-a-reasonable-price lens. Its job is to answer: **would this company look attractive through a Buffett-style investment philosophy?**

It emphasizes business quality, financial resilience, durable earnings power, shareholder friendliness, and intrinsic value. This agent is less about short-term trading and more about whether the company resembles a long-term compounder with a margin of safety.

### What it actually does

- **Input data:** full financials, line items, market cap, news, and insider trades.
- **Core method:** ten Buffett-style scores:
    - ROE.
    - Margins.
    - Debt/equity.
    - Current ratio.
    - Margin stability.
    - Capital intensity.
    - Book-value growth.
    - Share dilution.
    - Free-cash-flow conversion.
    - Intrinsic value.
- **Weighting:** intrinsic value is weighted heaviest.
- **LLM refinement:** an LLM refines the normalized score using a Buffett-philosophy prompt.
- **Decision rule:**
    - Bullish if normalized score ≥ 0.70.
    - Bearish if normalized score ≤ 0.30.
    - Neutral otherwise.
- **Output:** Buffett-style bullish / bearish / neutral signal refined by the LLM.

---

## Part 2 — Decision layer

The decision layer converts individual agent views into portfolio-ready outputs. It does three things: builds consensus, limits risk, and decides what the portfolio should do.

## 11. Consensus + STARS Rating · `aggregator`

### What this agent is meant to do

The Aggregator is Nova's voting and rating engine. Its job is to answer: **after all usable agents have spoken, what is the combined recommendation?**

It creates a single consensus from many agent opinions. Importantly, failed or abstained agents are excluded rather than counted as neutral, so the rating is based only on agents that produced usable outputs.

### What it actually does

- **Input data:** all available agent signals for a ticker.
- **Exclusion rule:** failed or abstained agents are excluded from the math.
- **Weighted score:** average of:
    - `+confidence` for bullish agents.
    - `-confidence` for bearish agents.
    - `0` for neutral agents.
- **Bounds:** score is bounded between -1 and +1.
- **Direction rule:**
    - Bullish if score > +0.10.
    - Bearish if score < -0.10.
    - Neutral otherwise.
- **STARS rating:**
    - **5 stars:** Strong Buy, score ≥ +0.40.
    - **4 stars:** Buy, score ≥ +0.10.
    - **3 stars:** Hold.
    - **2 stars:** Sell, score ≤ -0.10.
    - **1 star:** Strong Sell, score ≤ -0.40.
- **Output:** consensus direction, score, and 1–5 STARS recommendation.

---

## 12. Risk Manager · `risk`

### What this agent is meant to do

The Risk Manager is Nova's position-sizing guardrail. Its job is to answer: **how much exposure is safe for each ticker?**

It prevents a strong signal from becoming an oversized or overly correlated position. Instead of deciding whether the stock is attractive, it defines the maximum exposure allowed based on volatility and portfolio correlation.

### What it actually does

- **Input data:** price history across tickers.
- **Core method:**
    - Calculates annualized volatility per ticker.
    - Builds a correlation matrix.
    - Creates a volatility-adjusted base position limit.
    - Applies a correlation multiplier.
- **Correlation logic:** low-correlation names can receive hedging credit.
- **Output:** per-ticker position caps, including max dollars, max shares, and remaining room.

---

## 13. Portfolio Manager · `portfolio`

### What this agent is meant to do

The Portfolio Manager is Nova's execution-decision layer. Its job is to answer: **given the consensus and risk limits, what should the portfolio actually do?**

It translates research into buy, sell, hold, quantity, and hedge plans. This is where the research system becomes action-oriented while still respecting the risk manager's constraints.

### What it actually does

- **Input data:** consensus signals and risk limits.
- **Long-only / research behavior:**
    - Bullish → buy, sized by confidence within the risk cap.
    - Bearish → sell any existing long.
    - Neutral → hold.
    - Shorts are blocked in research mode.
- **Long/short behavior:**
    - Pairs the strongest bull with the strongest bear.
    - Sizes the hedge.
- **Output:** per-ticker buy / sell / hold action, quantity, and hedge plan.

---

## Part 3 — Debate engine

The Debate engine is Nova's LLM research pod. It is built for deeper thesis work where evidence quality, recency, counterarguments, and narrative synthesis matter more than a single numeric score.

## 14. Supervisor · planner

### What this agent is meant to do

The Supervisor is Nova's research coordinator. Its job is to answer: **what questions should the research pod investigate, and who should investigate each part?**

It turns the user's investment question into a structured research plan. Instead of every specialist looking at the same thing, the Supervisor splits the problem into focused mandates.

### What it actually does

- **Input:** the user's investment question.
- **Core method:** restates the question and splits it into four specialist mandates:
    - Fundamental.
    - Sentiment.
    - Valuation.
    - Macro.
- **Model:** Claude Opus.
- **Output:** research plan and specialist assignments.

---

## 15. Four Specialists · fundamental / sentiment / valuation / macro

### What these agents are meant to do

The Specialists are Nova's parallel research analysts. Their job is to answer: **what does the evidence say from each major investment lens?**

Each specialist owns a specific research angle, gathers evidence, forms a stance, and reports key findings with sources and confidence. Running them in parallel makes the research pod faster and helps avoid one blended, shallow answer.

### What they actually do

- **Execution:** run in parallel.
- **Evidence sources:**
    - SEC filings through BM25 RAG with cited chunk IDs.
    - Live web search.
- **Output structure:**
    - Stance.
    - Key findings.
    - Evidence.
    - Source.
    - Confidence.
- **Models:**
    - Claude Sonnet by default.
    - Claude Haiku in fast / quick tiers.
- **Mandates:**
    - **Fundamental:** financials, revenue, margins, guidance, balance sheet, and segments.
    - **Sentiment:** news flow, analyst rating changes, management tone, and market narrative.
    - **Valuation:** multiples versus history and peers, what's priced in, and upside/downside scenarios.
    - **Macro:** sector tailwinds/headwinds, competition, regulation, and demand.

---

## 16. Bear Challenger

### What this agent is meant to do

The Bear Challenger is Nova's adversarial reviewer. Its job is to answer: **what could break the thesis?**

It actively hunts for disconfirming evidence instead of reinforcing the base case. This is important because investment research is vulnerable to confirmation bias. The Bear Challenger makes sure demand risk, competition, valuation risk, concentration, and other severe risks are surfaced before the final memo.

### What it actually does

- **Core method:** attacks the thesis using disconfirming evidence.
- **Risk areas:** demand softening, competition, valuation risk, concentration, and other thesis-breaking issues.
- **Output:** refutations of specific claims with severity ratings.
- **Search behavior:** can use live web search; web search is off in quick tier.
- **Model:** Claude Opus by default, Claude Sonnet in quick tier.

---

## 17. Synthesizer · PM

### What this agent is meant to do

The Synthesizer is Nova's final investment memo writer. Its job is to answer: **after considering all evidence and the bear case, what should we believe?**

It does not simply average all opinions. It judges evidence quality, recency, and whether the bear case has created an unrebutted high-severity risk. The output is a portfolio-manager-style memo that explains conviction, lean, scenarios, risks, and what would change the view.

### What it actually does

- **Input data:** specialist drafts and bear-case review.
- **Core method:**
    - Reconciles all analyst views.
    - Judges evidence by quality and recency.
    - Does not average the views mechanically.
- **Conviction rule:** high conviction requires that the bear case surfaced no high-severity unrebutted risk.
- **Output memo includes:**
    - Conviction: high / medium / low.
    - Directional lean: constructive / neutral / cautious.
    - Bull narrative.
    - Base narrative.
    - Bear narrative.
    - Key risks.
    - What would change the view.
    - Citations.
- **Model:** Claude Opus.

---

## How to read Nova outputs

When Nova produces a ticker recommendation, read the output in this order:

**Note:** STARS, consensus, valuation target, and risk cap come from a Signals run. Bear Challenger findings and the Synthesizer memo come from a Debate run. These are complementary modes: you can read either one on its own, or compare both side by side.

1. **STARS rating** — the quick summary of the aggregate signal.
2. **Consensus score** — the strength and direction behind the rating.
3. **Valuation target** — whether the market price leaves enough upside/downside.
4. **Risk cap** — how much exposure is allowed even if the thesis is strong.
5. **Bear Challenger findings** — the most important counterarguments.
6. **Synthesizer memo** — the final research narrative and conviction.

## Design principles

- **Separate lenses:** each agent owns a specific job.
- **Do not average away risk:** the Synthesizer can override simple consensus if evidence quality or bear risks demand it.
- **Exclude failed agents:** unavailable agents do not dilute the score as neutral.
- **Use primary sources where possible:** SEC filings are the most authoritative evidence layer.
- **Keep portfolio logic separate from research logic:** a strong thesis still needs risk sizing.
- **Make outputs explainable:** every direction should include confidence, factors, and citations where available.
