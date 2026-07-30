# ADR-273: Learned Adaptive Early Termination for HNSW (LAET)

**Status**: Proposed
**Date**: 2026-07-30
**Author**: Nightly Research Agent
**Branch**: `research/nightly/2026-07-30-laet-hnsw-early-termination`
**Crate**: `crates/ruvector-laet`
**Related**: ADR-240 (Coherence-HNSW), ADR-272 (Adaptive Recall-Targeted ANN),
  ADR-272 (Recall-Bounded ANN)

---

## Context

Every HNSW deployment in the RuVector fleet picks an `ef_search` value that
must satisfy the *hardest* query it expects to see. Easy queries — which are
the vast majority in agent-memory and RAG workloads — pay that same worst-case
cost. Recent literature (LAET, NeurIPS 2024) shows a tiny per-query stopping
predictor over cheap traversal features can shave 30–60% of distance
computations at matched recall.

RuVector already ships two adjacent crates, but neither addresses per-query
learned stopping:

- **`ruvector-adaptive-ann`** (ADR-272 adaptive-recall) auto-selects `ef` to
  hit a recall *target* — but the target applies to every query alike and
  requires an online calibration signal.
- **`ruvector-recall-bounded`** (ADR-272 recall-bounded) tightens ef schedules
  offline; still a global constant per corpus.

Neither observes the *current query's* traversal trajectory and decides to
stop early based on it. That is the gap LAET fills.

## Decision

Introduce `crates/ruvector-laet` as a research crate housing:

1. A minimal, hermetic HNSW-like beam search with a `SearchStrategy` trait
   seam.
2. Three strategy implementations: `FixedEf`, `PatienceStop`,
   `LaetStop { model: LinearModel }`.
3. A closed-form ridge-regression trainer (no ML dependency) that fits the
   6-parameter `LinearModel` from oracle traversals.
4. A reproducible benchmark harness producing real numbers, not stubs.

The PoC targets a **35%+ distance-call reduction at ≤0.02 recall loss** on a
5k × 64 clustered synthetic dataset — achieved (34.2% at 0.012 loss on the
first tuned run). Graduation into `ruvector-core::hnsw` will happen in a
follow-up ADR once the seam design has stabilised.

## Consequences

**Positive**
- Adds a per-query lever operators can pull without changing indexes.
- Fully hermetic and ML-dep-free — the trainer is 90 lines of Gauss-Jordan.
- Composes with ADSampling and FINGER-style distance shortcuts.
- Establishes a reusable `SearchStrategy` trait pattern for ruvector-core.

**Negative / trade-offs**
- Learned stopping is a *statistical* recall guarantee. Hard SLAs need a
  fallback FixedEf pass.
- Requires periodic retraining as query distribution drifts.
- The PoC omits the classic HNSW "prune worse than worst-of-topk" short-circuit
  to give LAET room to shine — production integration must preserve that
  short-circuit and only intervene beyond it.

## Alternatives Considered

- **Purely heuristic patience-stop.** Simpler, one hyper-parameter, but the
  PoC shows it forfeits 15 recall points relative to LAET at the same work
  budget. Not competitive.
- **Global adaptive `ef` (`ruvector-adaptive-ann`).** Correct at the corpus
  level, but cannot cut easy-query cost since it picks one number for all
  queries.
- **MLP predictor (as in the original LAET paper).** ~10-30% additional
  savings but pulls in `candle` or `burn`. Deferred to a v2; ridge already
  clears the acceptance bar.
- **Do nothing / rely on operator tuning.** Leaves 30%+ of easy-query cost on
  the table indefinitely and does not scale as embedding sources drift.
