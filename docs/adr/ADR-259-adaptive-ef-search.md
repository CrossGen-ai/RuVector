---
adr: 259
title: "Per-Query Adaptive `ef`-search for HNSW"
status: proposed
date: 2026-06-18
authors: [nightly-research]
related: [ADR-001, ADR-253, ADR-254]
tags: [ruvector, hnsw, adaptive, ef-search, recall, latency, learned-index, laet, auncel]
---

# ADR-259 — Per-Query Adaptive `ef`-search for HNSW

## Status

**Proposed.**  PoC delivered as `crates/ruvector-adaptive-ef` on branch
`research/nightly/2026-06-18-adaptive-ef-search`.  Numbers measured;
integration into `ruvector-core` deferred to a follow-up.

## Context

HNSW exposes `ef_search` as the dominant latency/recall knob.  Every
shipped HNSW implementation in the ruvector workspace — including
`ruvector-core`, `ruvector-hyperbolic-hnsw`, and the WASM builds —
treats it as a **global, index-level constant**.

This is provably suboptimal.  Query difficulty is non-uniform:

* In-distribution queries inside dense clusters saturate recall at
  small `ef`.
* Queries near cluster boundaries require larger `ef` for the same
  recall.
* Out-of-distribution (OOD) queries benefit from `ef` an order of
  magnitude higher.

A single global `ef` chosen at deploy time either over-spends on easy
queries or under-recalls on hard ones.

Two SOTA lines of work address this:

* **LAET** (Li & Xu, SIGMOD 2020) — learns per-query `ef` from local
  neighbourhood-density features.
* **Auncel** (Wang et al., SIGMOD 2023) — Bing-production system that
  uses a tiny regressor for early-termination decisions, cutting p99
  latency 2–3× at fixed recall.

Neither is implemented in any current vector database (verified against
Milvus, Qdrant, Weaviate, Pinecone, LanceDB, FAISS 2024–2026 changelogs).
This is a green-field opportunity for ruvector.

## Decision

**Adopt a swappable `EfPredictor` trait as the unit of `ef`-selection**,
and ship three concrete implementations in `ruvector-adaptive-ef`:

1. `FixedEf` — backwards-compatible baseline.
2. `HeuristicAdaptiveEf` — LAET-style rule-based predictor; no training.
3. `LearnedAdaptiveEf` — closed-form ridge regression on
   `(features → min_ef_for_target_recall)` pairs (5 floats, edge-friendly).

The features are 4 cheap statistics computed from `query × 8 pivots`:
`min_pivot_d2`, `mean_pivot_d2`, `std_pivot_d2`, `min_over_mean`.  All
predictors implement `EfPredictor::predict(&QueryFeatures) -> usize`.

**Integration plan (future work, not part of this ADR):** add an
`EfPredictor` parameter to the production HNSW search path in
`ruvector-core`.  Default `FixedEf(ef_search)` preserves bit-for-bit
backwards compatibility; opt-in `LearnedAdaptiveEf` behind a
`adaptive-ef` cargo feature.

## Consequences

### Positive

* **Continuous recall control.**  One deployed model covers the whole
  Pareto front; no per-target re-tune.
* **Honest mean-cost wins.**  At recall ≈ 0.88 on heterogeneous
  10K × 64d synthetic data, `LearnedAdaptiveEf` matches `FixedEf(128)`
  recall at `FixedEf(96)` cost (892 vs 988 mean distance evaluations).
* **Small footprint.**  Model = 5 `f32`s ≈ 20 bytes; predictor overhead
  ~8 dot products / query (< 1 µs).
* **Drop-in trait.**  Composes with existing HNSW search; no graph-build
  changes required.
* **Foundation for SLA work.**  Predictor outputs are inspectable —
  `ef_max` clamping enforces hard p99 budgets.

### Negative

* **p99 inflation.**  Hard queries get clamped to `ef_max`, raising p99
  distance evaluations vs `FixedEf(96)`.  This is the intended trade
  for mean-cost wins, but it bites SLA budgets that price p99.
* **Pivot drift sensitivity.**  Predictor accuracy degrades if the
  dataset distribution drifts away from the pivot set.  Requires
  drift monitoring (PSI/KS on feature histograms).
* **Cold-start cost.**  Label generation requires brute-force k-NN +
  `ef` sweep on a training query set.  ~1 sec / 500 queries on 10K data;
  scales linearly.
* **Synthetic-data win is modest.**  On uniform-cluster MoG, learned
  doesn't strictly dominate the full fixed-`ef` Pareto front; real
  embeddings are expected to show larger wins (per Auncel) but we have
  not yet measured this in ruvector.

### Neutral

* The crate ships a brute-force-built `MiniHnsw` for reproducible
  benchmarking, **not** as a replacement for production HNSW.

## Alternatives considered

### A. Do nothing (status quo)

Continue with `FixedEf` only.  Rejected: leaves a measurable
optimisation on the table; no Rust-ecosystem alternative exists.

### B. Per-collection `ef` only

Allow `ef_search` per collection (already supported by Qdrant).
Rejected: doesn't address intra-collection query-difficulty variance,
which is where Auncel's measured wins come from.

### C. K-only adaptive (Weaviate-style)

Auto-tune `ef = max(k * α, ef_min)`.  Rejected: ignores per-query
difficulty entirely; equivalent to interpolating on the fixed-`ef`
sweep, not improving over it.

### D. Online RL controller

Train a contextual bandit that adapts `ef` based on observed recall.
Rejected for v1: requires online recall estimation, which itself
requires brute-force probes.  Complexity-to-payoff ratio worse than the
offline-labelled regressor for first delivery.

### E. Bigger learned model (MLP)

Use a 2-layer MLP per query.  Rejected for v1: predictor latency floor
matters (the whole point is to *save* µs per query).  Ridge regression
is the right size; revisit if features grow past ~20.

## Measurement

```bash
cargo test  -p ruvector-adaptive-ef --release  # 9/9 pass
cargo run   -p ruvector-adaptive-ef --release --bin adaptive-ef-bench
```

Full numbers in
[`docs/research/nightly/2026-06-18-adaptive-ef-search/README.md`](../research/nightly/2026-06-18-adaptive-ef-search/README.md).
