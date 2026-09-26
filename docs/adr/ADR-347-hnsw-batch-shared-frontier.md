# ADR-347: Batched HNSW Search with Shared Candidate Frontier

## Status

Proposed. Standalone PoC in `crates/ruvector-hnsw-batch-shared-frontier`,
opt-in via crate selection. Non-breaking (no existing crate depends on it).

## Context

Existing HNSW-family crates in the workspace
(`ruvector-hnsw-prefetch`, `ruvector-coherence-hnsw`,
`ruvector-bidir-hnsw`, `ruvector-graph-locality-hnsw`,
`ruvector-hnsw-repair`, `ruvector-hnsw-delta-neighbors`,
`ruvector-hnsw-rng-diverse`, `ruvector-ef-neighbors`) all optimise the
**single-query** greedy-search pathway. In many ruvector workloads,
however, queries arrive in **correlated batches**:

- Multi-turn RAG within a session
- MMR / diverse-beam expansion on a single seed query
  (`ruvector-diverse-beam`, `ruvector-speculative-ann`)
- Query rewrite fan-out (paraphrase, expansion)
- Multi-vector retrieval / MaxSim-style scoring
  (`ruvector-maxsim`, `ruvector-muvera`)

The prior nightly `2026-09-25-ef-hnsw-neighbors` tuned per-query `ef`,
but each query still traverses the graph independently, re-loading every
candidate vector's cache lines for every query. On a correlated batch
the visited-node overlap is high — 80-97% at cosine correlation ≥ 0.85
on the reference dataset — so most of those loads repay no new
information.

## Hypothesis

Given a batch of `Q` correlated queries and an HNSW-style proximity
graph, if greedy `ef`-search is refactored so that

1. distance computations use a **batched kernel** (one candidate vector
   × Q queries per call, keeping the candidate hot across queries), and
2. a **batch-wide evaluated set** caches the Q-length distance row of
   every node the batch has already visited, so later queries reuse it,

then the number of distinct candidate vector loads drops by 70-97 % on
correlated batches, wall-clock latency drops by 10-30 % on Apple M4 Max,
and per-query recall remains **numerically identical** to the
independent baseline (each query still owns its own frontier heap and
result heap).

## Decision

Publish `crates/ruvector-hnsw-batch-shared-frontier` implementing three
swappable backends behind the `BatchedSearch` trait:

- `IndependentSearch` — baseline; per-query greedy search.
- `BatchedKernelSearch` — batched distance kernel only.
- `SharedFrontierSearch` — batched kernel + evaluated-set cache.

Downstream crates that already batch queries (`ruvector-maxsim`,
`ruvector-muvera`, `ruvector-diverse-beam`, `ruvector-bounded-rag`)
gain a drop-in `BatchedSearch` trait to switch backends per workload.

## Consequences

### Positive

- **Vector-load savings 71-97 %** (measured, `crates/ruvector-hnsw-batch-shared-frontier` bench binary, Apple M4 Max, n=5000, dim=128, ef=64).
- **Wall-clock 10-30 % faster** than the independent baseline at Q ≥ 8, correlation ≥ 0.85.
- **Recall preserved bit-for-bit** vs baseline — the shared cache changes
  *which* memory holds a distance value, not the value itself. Backed by
  a differential test (`shared_saves_loads_on_correlated_batch`).
- **Memory-bounded cache**: `Q · N · 4` bytes worst case
  (~0.6 MiB at Q=32, N=5000; scales linearly).

### Negative / Costs

- `BatchedKernelSearch` on its own is **slower** than the independent
  baseline (3-15 ×) because it does Q distance computations per visited
  node even when only one query progresses. It is retained as an
  ablation and as a building block for the shared variant; production
  paths should skip straight to `SharedFrontierSearch`.
- Shared-cache footprint scales with `Q · N`; for very large `N` (e.g.
  ≥ 10 M vectors) a sparse (hash-map-backed) cache is required. The PoC
  uses a dense `Vec<f32>` and documents the swap point.
- Correlated-batch assumption. On **uncorrelated** batches the overlap
  vanishes and the shared cache degenerates to the batched-kernel cost.
  Callers should route uncorrelated batches to `IndependentSearch`.

### Non-goals

- Full HNSW multi-layer implementation. The PoC uses a single-layer
  proximity graph so the experimental variable is the *search access
  pattern*, not build quality. Production integration would slot the
  same three backends into `ruvector-coherence-hnsw`'s layer-0 search.
- Query clustering to *induce* correlation on non-correlated batches;
  that is a separate follow-up (see research doc "What to improve next").

## Alternatives Considered

- **Merged-frontier search** (single heap over the union of queries).
  Rejected: destroys per-query top-k semantics and complicates recall
  accounting.
- **Query centroid + rerank** (search the batch centroid once, then
  rerank candidates against each query with the batched kernel).
  Cheaper but recall drops sharply for correlation < 0.9. Retained as
  a follow-up experiment.
- **GPU batched search.** Out of scope for a CPU-first Rust nightly;
  the batched-kernel + shared-cache abstraction is a prerequisite for
  a GPU port anyway.

## Acceptance Criteria

- `cargo build --release -p ruvector-hnsw-batch-shared-frontier` passes.
- `cargo test  --release -p ruvector-hnsw-batch-shared-frontier` passes
  (6 tests, including a differential vector-load-count assertion).
- Bench binary reports ≥ 70 % vector-load savings and non-degraded
  recall at Q ≥ 4, correlation ≥ 0.6, on the reference dataset.
- All three backends satisfy `BatchedSearch` and return identical top-k
  results on the same seed.
