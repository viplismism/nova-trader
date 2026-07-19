# AlphaDesk analyst trust audit

## Purpose

This audit classifies every analyst by what it reads, how it reaches a result, whether an LLM controls the result, whether the evidence overlaps another analyst, and whether it should participate in the core consensus.

The classification is about current implementation quality, not whether the underlying financial idea is legitimate.

## Trust categories

- **Direct:** based on market prices, statements, transactions, or primary filings.
- **Derived:** deterministic finance/math applied to direct inputs.
- **Heuristic:** simplified rules or lexical counts used as a proxy.
- **LLM-authored:** a language model controls final stance or confidence.

## Analyst matrix

| Analyst | Input provenance | Final result | Main overlap | Current risk | Recommendation |
|---|---|---|---|---|---|
| Fundamentals | Direct financial metrics | Deterministic threshold score | Growth, Valuation, Buffett | Missing values fail checks; fixed sector-agnostic thresholds | **Core after fixes** |
| Growth | Direct multi-period metrics | Deterministic weighted score | Fundamentals, Valuation | Historical trend chronology appears reversed; missing fields score zero | **Core after chronology fix** |
| Valuation | Direct financials plus model assumptions | Deterministic four-model blend | Fundamentals, Growth, Buffett | WACC debt-cost bug; missing net-debt/WC inputs; false precision | **Core after model fixes and sensitivity output** |
| Technical | Direct price/volume | Deterministic five-strategy heuristic | Risk uses same prices for a different purpose | No minimum history; neutral dilution; several approximate indicators | **Separate timing signal** |
| SEC Filings | Primary SEC passages | Deterministic keyword heuristic | Adaptive Research, Debate | Excellent evidence source but weak semantic vote | **Evidence only** |
| Web Research | Search titles/snippets | Deterministic keyword heuristic | News, Adaptive Research, Debate | Duplicate/low-quality sources; no semantic verification | **Evidence only** |
| News Sentiment | Provider labels or headline keywords | Deterministic classifier | Web Research | Headline-only; no deduplication or source weighting | **Evidence only** |
| Insider Sentiment | Reported insider transactions | Deterministic count heuristic | Buffett view receives same data but barely uses it | Ignores size and transaction reason; tiny samples reach 95% | **Evidence/risk flag only** |
| Warren Buffett | Direct financials transformed into scorecard | **LLM controls final signal** | Fundamentals and Valuation | LLM can override deterministic hint; missing data penalized; persona label overstates authority | **Remove from default; optional persona** |
| Social Sentiment | Reddit/community text | Deterministic VADER heuristic | News/Web narrative | One viral post can dominate; comments mixed into post; weak causal value | **Optional context only** |
| Adaptive Research | Web and SEC evidence | **LLM controls final signal** | Web, SEC, Debate | Model-authored confidence; overlapping evidence becomes another vote | **Move to deep research; no core vote** |

## Decision-layer audit

| Component | Current behavior | Risk | Recommendation |
|---|---|---|---|
| Consensus score | Signed analyst confidence divided by participant count | Treats incompatible confidence scales as comparable; correlated evidence votes repeatedly | Aggregate only calibrated core models; expose raw component scores |
| Consensus confidence | Average confidence regardless of direction | Can be high when analysts strongly disagree | Rename immediately; redesign as agreement × evidence quality × coverage |
| STARS | Label mapped from consensus score | Exact ±0.10 boundary differs from direction boundary | Align thresholds and document as a display label |
| Risk Manager | Volatility/correlation heuristic | Formula discontinuities and comments disagree with implementation | Fix boundaries; test every regime; expose inputs |
| Portfolio Manager | Converts consensus and limit into action | Uses misleading consensus confidence for sizing; incomplete sell/short logic | Do not execute live; finish and test state transitions first |

## Validation status

The system has strong run-level traceability but incomplete model-level validation.

### Present

- Saved input snapshot and per-agent views.
- Saved signals and final recommendation.
- Saved LLM prompts/responses.
- Structured source citations for filings/web evidence.
- Broad integration and UI test coverage.

### Missing or incomplete

- Dedicated threshold tests for Fundamentals, Growth, Valuation, Technical, Insider, and Buffett runners.
- Confidence calibration against outcomes.
- Per-source freshness and quality score.
- Valuation sensitivity tables and invariant tests.
- Missing-data contracts distinguishing “bad” from “unknown.”
- Independence/correlation analysis between analyst votes.
- A frozen methodology version attached to each displayed score.
- Decision-quality evaluation by market regime and investment horizon.

## Recommended target

Do not delete useful data collection. Delete the assumption that every collected signal deserves an equal vote.

The recommended target is:

```text
CORE BUSINESS SCORE
  fundamentals + growth + valuation

SEPARATE MARKET-TIMING SCORE
  technical

EVIDENCE AND RISK FLAGS
  SEC + web + news + insider + social

EXPLICIT DEEP RESEARCH
  adaptive + Buffett persona + debate
```

Before changing the production council, agree with Deep on:

1. the intended investment horizon;
2. which outputs are facts, scores, or opinions;
3. whether Technical affects recommendation or only timing;
4. how missing data should affect a score;
5. which valuation assumptions are defensible;
6. how confidence will be calibrated; and
7. which historical benchmark will validate usefulness.
