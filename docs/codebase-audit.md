# AlphaDesk codebase audit

## Summary

The runtime has a strong typed core—snapshot, typed views, standard signals, consensus, risk, recommendation, and persisted artifacts—but several generations of product work coexist in the same package. The result is more surface area and duplicated concepts than the active production path requires.

This should be cleaned incrementally, with behavior-locking tests before methodology changes.

## Safe cleanup completed with this audit

- The normal default analyst set now lives beside the registry instead of being duplicated in the chat layer.
- Dead Valuation parameters and unused scenario configuration were removed without changing results.
- An unused Fundamentals helper was removed.
- Stale `v2`/legacy ownership comments in active math modules were replaced with their current owners.
- Registry behavior now has a direct invariant test.

## Highest-risk findings

### 1. Methodology has insufficient unit coverage

Several core analyst runners do not have direct tests around thresholds, missing data, and invariants. Integration tests prove the application runs; they do not prove the financial formulas are correct.

Required first: characterization tests for every analyst and exact boundary.

### 2. Multiple concepts have more than one source of truth

- Registered and default analysts previously had separate definitions; this was consolidated into `src/registry.py` during this audit.
- Canonical recommendation schemas live in `src/schemas/signals.py`; a second recommendation model family lives under `src/recommender/` and is not used by the active engine.
- The active chat router uses `src/chat/routing.py`; a separate generic router contract lives under `src/router/`.
- The active portfolio decision path produces typed `Decisions`; a separate execution bridge expects legacy dictionaries.
- LLM configuration is split between the active `src/utils/llm.py` adapters and an older model-catalog layer under `src/llm/`.

Each concept needs an explicit owner; unused scaffolding should move out of the production package or be deleted after dependency verification.

### 3. Large modules mix responsibilities

- `src/chat_cli.py` exceeds 2,000 lines and combines application state, rendering, routing, run orchestration, saved-run inspection, and debate UI.
- `src/web/server.py` combines authentication, Signals execution, Debate execution, persistence recovery, question answering, and HTTP route definitions.
- `src/utils/llm.py` combines provider selection, credentials, retries, structured output, streaming, fallback policy, and provider-specific adapters.
- `src/debate/engine.py` combines schemas, prompts, provider tools, orchestration, fallback behavior, and synthesis.

These should be split by capability only after tests protect their public behavior.

### 4. Broad exception handling hides state quality

Best-effort collection is appropriate at I/O boundaries, but broad `except Exception` blocks are also used across model and orchestration code. The system often degrades to empty data, neutral, or fallback output without a typed reason that the decision layer can weigh.

Recommended: typed data-quality states such as `fresh`, `fallback`, `stale`, `partial`, and `unavailable`, persisted with each view and signal.

### 5. Formula wrappers and math modules have unclear ownership

Current analyst runners live in `src/agents/*.py`, while core calculations live in `src/agents/math/*.py`. This can be a good separation, but comments still refer to removed `v2` and legacy paths, and dead parameters remain in Valuation. The public API and ownership boundary are not documented or enforced.

Recommended convention:

```text
src/agents/<name>.py          orchestration and Signal construction
src/domain/<name>.py          pure, independently tested formulas
tests/domain/test_<name>.py   formula and invariant tests
```

### 6. Presentation logic is duplicated across surfaces

Recommendation interpretation and formatting are spread across web JavaScript, `src/chat/signal_card.py`, `src/chat/rendering.py`, `src/chat_cli.py`, and CLI display helpers.

Recommended: one presentation DTO and one vocabulary module; each UI should only render that DTO.

## Cleanup sequence

### Phase 1 — protect behavior

- Add direct unit tests for all core analyst formulas and edge cases.
- Add golden tests for consensus, risk regimes, and portfolio state transitions.
- Add data-quality fixtures for primary, fallback, partial, and missing sources.

### Phase 2 — establish ownership

- Make the registry the single source of truth for registered/default analysts.
- Mark unused recommendation/router/execution scaffolds as experimental or remove them.
- Define one canonical model/provider configuration source.
- Define one presentation DTO consumed by web and terminal interfaces.

### Phase 3 — split orchestration

- Split web auth, Signals routes, Debate routes, and run-query routes.
- Split chat state, commands, rendering, and inspectors.
- Split LLM provider adapters from retry/fallback policy.
- Split Debate schemas/prompts from execution.

### Phase 4 — improve methodology

- Fix Growth chronology and Valuation assumptions.
- Separate evidence collectors from consensus voters.
- Redesign confidence and calibration.
- Version every methodology and persist that version with runs.

### Phase 5 — remove compatibility debt

- Retire legacy dictionary adapters after backtesting/execution consumers migrate.
- Remove orphaned schemas and model catalogs.
- Remove stale generated methodology artifacts or regenerate them from one source.

## Senior-engineering rule for this cleanup

Do not combine a large structural refactor with a methodology change in the same commit. First lock behavior with tests, then refactor without changing outputs, then change one financial rule at a time with explicit before/after fixtures.
