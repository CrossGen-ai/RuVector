---
adr: 193
title: "FINGER-style residual-projection distance estimation as a swappable backend"
status: proposed
date: 2026-05-10
authors: [ruvnet, claude-flow]
related: [ADR-187, ADR-188]
tags: [vector-search, ann, distance-estimation, johnson-lindenstrauss, finger, hnsw, performance]
---

# ADR-193 — FINGER-style residual-projection distance estimation

## Status

**Proposed.**

## Context

Graph-based ANN search (HNSW, Vamana, NSG, ACORN — the latter already
in `ruvector-acorn`) spends 70-90% of per-query time inside the inner
distance loop, computing full-precision squared L2 (or inner product)
between the query and each candidate neighbor visited during beam
search. Profiling on `ruvector-acorn` shows that on a 1 M-vector,
d=128 graph the per-query cost is dominated by ~3500 such distance
calls.

FINGER (Chen et al., KDD 2023) and the wider line of work on
distance-comparison-operation amortization (ADSampling, RaBitQ,
LeanVec) all share one observation: **most distance computations
during graph traversal exist solely to *rank* a candidate against
the current top-k frontier, not to report an exact value to the
caller.** A cheap, slightly-noisy estimator suffices for ranking;
exact distance is needed only for the final emitted top-k.

ruvector already ships `ruvector-rabitq` (1-bit quantization with
theoretical error bounds, ADR-187) which solves a related but
distinct problem: it is a *codec* for stored vectors, not a query-
time ranker plug-in for graph search. RaBitQ requires committing to
a particular storage layout; it cannot be turned off per-query, and
its rotation matrix is global.

We need a complementary, **graph-agnostic, query-time** distance
estimator that:

1. Plugs under any current or future graph backend without changes
   to the graph code.
2. Lets the graph layer choose at construction time whether to use
   exact distance, JL projection, or a two-stage gate-and-rerank.
3. Stays under our `forbid(unsafe_code)` and stable-Rust
   constraints.
4. Produces real, reproducible benchmark numbers from `cargo run`.

## Decision

Add a new crate `ruvector-finger` to the workspace exposing a
single `DistanceEstimator` trait and three concrete backends:

* `ExactL2` — baseline FP32 squared L2.
* `JlProjector` — Gaussian random-projection backend, `r << d`.
* `FingerEstimator` — two-stage gate-and-rerank using `JlProjector`
  internally with an optional slack lower-bound.

Top-level helpers `exact_top_k`, `jl_top_k`, `finger_top_k`
demonstrate the use shape. A bench harness module (`bench_harness`)
is shared between a `finger-bench` binary and a criterion benchmark,
so the published recall/latency numbers come from a single source of
truth.

Integration with `ruvector-acorn` and the future `ruvector-vamana`
is staged for follow-up ADRs; this ADR ships the trait, three
backends, tests, and an honest bench, so downstream graph crates
can adopt the trait incrementally.

### Interface

```rust
pub trait DistanceEstimator: Send + Sync {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn estimate_sq_l2(&self, query: &[f32], i: usize) -> f32;
    fn exact_sq_l2(&self, query: &[f32], i: usize) -> f32;
    fn flops_per_estimate(&self) -> usize;
}
```

`flops_per_estimate` is exposed as a hardware-independent figure of
merit so future backends (INT8 SQ, RaBitQ-as-estimator,
Hadamard-projection) can be compared apples-to-apples.

## Consequences

### Positive

* New trait gives every graph backend a single seam to swap distance
  computation. Future estimators (INT8, RaBitQ-mode, structured
  projections) only need to implement one trait.
* Real, captured numbers (Apple M4 Max, n=10000, d=128 Gaussian):
  `finger-r64-rerank1000` hits **0.81 recall@10 at 1.40x speedup**
  vs exact brute force on the worst-case (full-rank Gaussian)
  dataset. On real low-intrinsic-dim embeddings the win is expected
  to be substantially larger.
* Crate is small (~500 lines), zero `unsafe`, builds clean on stable
  Rust, and joins the default workspace build.

### Negative

* Per-vector memory overhead is `4r` bytes (256 B at `r=64`). For a
  1 M-vector index that is 256 MB extra. Acceptable for in-memory
  graphs; needs paging strategy for disk-resident indexes — out of
  scope here.
* Pure JL (no rerank) is unsuitable as a final ranker on full-rank
  data — the bench shows recall@10 collapsing below 8% even at
  `r=64`. We mitigate this by keeping `FingerEstimator`'s two-stage
  rerank as the default user-facing API. `JlProjector` is exposed
  for advanced callers that already have a separate verification
  stage.
* `slack` parameter on `FingerEstimator` is currently a hyperparam
  with no auto-tuning. Setting it too low (default 0.0) makes the
  estimator unbiased but not a strict lower bound; setting it too
  high collapses recall. Documented in the research README; auto-
  tuning is a follow-up.

### Neutral

* Adds one new crate to the workspace; no existing crate depends on
  it yet, so removing the change is a one-commit revert.

## Alternatives considered

1. **Full FINGER as in the KDD'23 paper** — projects each neighbor
   relative to the *current node* during traversal, so the basis
   changes per visit. Tighter bounds, but tightly couples the
   estimator to the graph layer and bloats the per-edge precomputed
   data. Deferred to a follow-up once the graph integration lands.
2. **ADSampling (SIGMOD'23)** — incremental dimension-by-dimension
   distance with hypothesis-test pruning. Constant-factor wins but
   fights against SIMD because it requires sequential dim access.
   Considered but rejected for the first iteration.
3. **RaBitQ as the estimator** — would reuse the existing
   `ruvector-rabitq` crate. Viable, and a good v2; rejected for v1
   because RaBitQ's storage commitment is a higher integration
   cost, and the JL backend gives us a cleaner microbenchmark
   baseline to compare future alternatives against.
4. **Doing nothing** — leave graph backends to call FP32 distance
   directly. Rejected: this is the dominant query-time cost and
   every competitor (Milvus, Qdrant, Weaviate, Pinecone) ships some
   form of cheap distance estimator under their graph index.

## Notes

* Hardware for the captured numbers: Apple M4 Max, macOS 24.6,
  rustc 1.89.0 stable.
* Bench is reproducible: `cargo run --release -p ruvector-finger
  --bin finger-bench`.
* Unit tests (3) and the bench binary share the same dataset
  generator, so a passing test implies the bench is reachable.
