# AlphaDesk methodology — the simple version

## Start here if you are technical, not financial

AlphaDesk should be understood as a typed decision pipeline, not as eleven intelligent people independently predicting a stock.

```text
one saved data snapshot
        ↓
specialized checks and evidence collectors
        ↓
standard Signal objects
        ↓
consensus
        ↓
risk limit
        ↓
buy / sell / hold
```

The full formula-level audit still exists for verification: [open the advanced audit](/methodology/audit).

---

## The most important correction

Most default analysts do **not** ask an LLM to invent their numbers. Eight of the nine default analysts are deterministic.

But deterministic does not automatically mean reliable. A fixed keyword counter is deterministic too. It gives the same answer twice, but the answer can still be shallow.

There are four levels of output in the current system:

| Level | Meaning | Examples |
|---|---|---|
| Direct fact | Copied or calculated directly from primary financial/market data | revenue, free cash flow, share price, market cap |
| Financial model | Formula applied to direct facts using visible assumptions | DCF, Owner Earnings, growth score |
| Heuristic | Simplified rule that approximates something more complicated | keyword counts, insider transaction counts |
| LLM judgment | A model reads evidence and authors the final stance/confidence | Buffett final signal, Adaptive Research, Debate |

The product should display these levels differently. Today they all become similar-looking analyst votes, which makes weak and strong evidence look more comparable than they really are.

---

## The current analysts, in technical language

### Fundamentals — latest-state health check

Reads direct financial ratios and runs fixed assertions over profitability, growth, debt, liquidity, and basic valuation.

```text
tech analogy: /health endpoint plus a linter
```

Useful as a decision model after missing-data handling and sector-specific thresholds are improved.

### Growth — historical trend monitor

Reads several financial periods and scores revenue, earnings, free cash flow, margins, valuation, and financial health.

```text
tech analogy: comparing service metrics across several releases
```

Useful as a decision model, but its chronology bug must be fixed before trusting the acceleration bonus.

### Valuation — estimation ensemble

Runs four visible models—DCF, Owner Earnings, EV/EBITDA, and Residual Income—and compares the blended company value with market cap.

```text
tech analogy: four independent cost/capacity estimators behind one interface
```

Useful as a decision model after its WACC and missing-input assumptions are corrected and sensitivity ranges are shown.

### Technical — price time-series monitor

Runs five deterministic detectors over price and volume. It knows nothing about company quality.

```text
tech analogy: runtime telemetry, not source-code inspection
```

Potentially useful as a separate timing signal. It should not be presented as equivalent to intrinsic business value.

### SEC Filings — primary-document evidence collector

Retrieves real 10-K/10-Q passages, then currently converts positive/risk word counts into a vote.

```text
tech analogy: searching official specifications and security disclosures
```

The source is excellent; the voting method is weak. Keep the passages and citations, but treat them as evidence rather than an independent recommendation.

### Web Research — external evidence collector

Searches current web titles/snippets and currently turns constructive/risk keyword counts into a vote.

```text
tech analogy: issue/search aggregation with a regex severity score
```

Useful for recency and citations, but better as evidence than as a consensus voter.

### News Sentiment — headline classifier

Classifies recent headlines using provider labels or a small keyword fallback.

```text
tech analogy: log classification from message titles only
```

Useful as context. Too shallow for an equally weighted investment vote without deduplication, credibility, recency, and article-level understanding.

### Insider Sentiment — behavioural event counter

Counts reported insider buys and sells. It does not distinguish transaction size, grants, tax sales, or planned sales.

```text
tech analogy: counting deploy events without inspecting payload size or reason
```

Keep as a risk/evidence flag, not a full vote.

### Warren Buffett — opinionated scorecard plus LLM

Builds a deterministic quality/value score, then allows an LLM to author the final direction and confidence.

```text
tech analogy: a policy engine whose final allow/deny can be overridden by a model
```

It overlaps Fundamentals and Valuation and should leave the default council. It can remain an optional persona view if clearly labelled.

### Social Sentiment — crowd telemetry

Scores Reddit/community language and weights Reddit posts by engagement.

```text
tech analogy: community mood monitoring
```

Useful as optional context, not a core investment vote.

### Adaptive Research — tool-using LLM researcher

An LLM plans searches, retrieves web/filing evidence, and authors a direction and confidence.

```text
tech analogy: an autonomous investigation agent
```

Useful in Debate or a research tab. It should not silently join a deterministic consensus.

---

## Proposed simpler architecture

### Decision core

These are candidates to produce scored signals after their known bugs are fixed:

```text
Fundamentals  → business health
Growth        → business trajectory
Valuation     → price versus modelled value
Technical     → market timing, shown separately
```

Technical should remain visibly separate from the three business-analysis models because it answers a different question.

### Evidence layer

These should collect and summarize facts without adding equal votes to consensus:

```text
SEC filings
web research
news
insider activity
social sentiment
```

### Deep-research layer

These can use LLM judgment, but their provenance must remain explicit:

```text
Adaptive Research
Buffett persona
Debate
```

They should inform a human or attach evidence—not quietly manufacture another precise percentage inside the deterministic score.

---

## Consensus in one minute

Each successful analyst currently contributes:

```text
bullish → +confidence
bearish → -confidence
neutral → 0
```

The signed total is divided by the number of participating analysts.

The displayed “consensus confidence” is a separate calculation: it is simply the average confidence reported by all participating analysts, including neutral analysts.

Therefore:

> 64% consensus confidence does not mean a 64% probability of being right. It currently means the average analyst self-reported confidence was 64%.

This metric should be renamed or redesigned.

---

## Risk and action

The Risk Manager does not decide whether the company is good. It uses price volatility, correlation, current exposure, and cash to set a maximum position.

```text
tech analogy: an SRE quota service
```

The Portfolio Manager translates direction plus the quota into buy, sell, or hold.

```text
tech analogy: a deployment controller
```

Research chooses the direction. Risk controls the size. Portfolio logic chooses the action.

---

## What is auditable today?

Every Signals run saves:

- the exact input snapshot;
- the exact typed view given to every analyst;
- every analyst signal;
- every LLM request/response;
- the consensus, limits, and final decision; and
- the source-code commit when available.

That gives us traceability. It does not yet give us validation.

Validation additionally requires:

- unit tests at every threshold;
- historical calibration of confidence;
- sensitivity tests for valuation assumptions;
- source-quality and freshness checks;
- missing-data tests;
- removal of correlated duplicate votes; and
- backtests that avoid look-ahead and survivorship bias.

---

## The five things to remember

1. Direct data, formulas, heuristics, and LLM judgments are different trust levels.
2. A deterministic number can still come from a weak rule.
3. Analyst confidence is not probability.
4. Evidence collectors should not automatically become equal decision voters.
5. The simpler target is three business models, one separate timing model, an evidence layer, and an explicitly labelled LLM research layer.

