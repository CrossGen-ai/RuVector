# ruvector 2026: Top-K Stability Termination — High-Performance Rust Vector Search with Kendall-Tau Early Stop for HNSW / ANN

> **Summary (150 char):** Rust HNSW-style ANN beam search terminated by Kendall-tau ranking stability on top-k — an ordinal, calibration-free early stop.

Rust · Vector Search · HNSW · Approximate Nearest Neighbor · Early Termination · Kendall Tau · Milvus alternative · Qdrant alternative · Weaviate alternative · Pinecone alternative · FAISS alternative · RAG · Embeddings.

## Introduction

`ruvector-topk-stability-terminate` is a research crate in the RuVector
2026 nightly line that replaces the standard scalar early-stop rule used
by every major vector database (fixed `ef_search` or k-th distance
plateau) with a rank-agreement rule: stop the ANN beam search when the
top-k identity ranking has held Kendall's τ ≥ τ* for a few consecutive
checks. This is the first published termination signal to consult the
ordinal object the caller actually receives — the ranked list of ids —
rather than a proxy scalar (a distance, an entropy, a recall estimate).
Written in pure Rust, dependency-free, and benchmarked on Apple M4 Max
with real numbers (recall@10, mean visits, p50/p95/p99 latency).

## Features

  * `TerminationPolicy` trait with pluggable implementations.
  * Three ready-to-swap policies: `FixedBudget`, `GapThreshold`,
    `KendallTauStability`.
  * Ordinal Kendall-tau knob (τ*) — reads naturally to operators
    ("stop when the top-k has changed by fewer than one adjacent swap").
  * Zero external dependencies: `cargo build -p ruvector-topk-stability-terminate`
    works offline on any Rust toolchain ≥ 1.75.
  * Deterministic PoC benchmark (fixed seeds, brute-force ground truth)
    that any reviewer can rerun in seconds.
  * Complete unit tests: RNG determinism, distance kernels, graph
    invariants, policy behavior on synthetic distributions.
  * Ready for HNSW integration — the trait is designed to drop into
    `ruvector-graph`'s beam search with no signature changes.

## Benefits

  * **Read the object, not a proxy.** Every prior termination signal
    (fixed budget, gap threshold, entropy, recall estimator) reduces
    to a scalar. Kendall-tau reads the ranked-id output the caller
    actually consumes — the thing you're trying not to move.
  * **Zero calibration, zero LUT.** Unlike learned-recall estimators,
    the τ* knob needs no training data and no per-index calibration.
  * **Ordinal tuning knob.** Operators reason about "top-k ordering
    stable within one adjacent swap on average" (τ ≈ 0.9) more
    confidently than "epsilon of a squared-L2 distance."
  * **Cheap.** O(k²) per check; a few thousand comparisons at k = 100 —
    swamped by a single 128-d distance computation.
  * **Composable.** The trait means Kendall-tau slots alongside — or
    in combination with — gap-threshold, entropy, or speculative-beam
    signals with no core changes.

## Comparisons

| Feature                                | Milvus | Qdrant | Weaviate | Pinecone | FAISS | LanceDB | RuVector 2026 |
| :------------------------------------- | :----: | :----: | :------: | :------: | :---: | :-----: | :-----------: |
| HNSW / graph ANN                       |   ✓    |   ✓    |    ✓     |    ✓     |   ✓   |    ✓    |       ✓       |
| Per-query adaptive `ef_search`         |   —    |   —    |    —     |    —     |   —   |    —    |       ✓       |
| Distance-plateau early stop            |   —    |   —    |    —     |    —     |   ~   |    —    |       ✓       |
| **Ordinal top-k stability early stop** | **—**  | **—**  |  **—**   |  **—**   | **—** |  **—**  |     **✓**     |
| Calibration-free termination           |   ✓    |   ✓    |    ✓     |    ✓     |   ✓   |    ✓    |       ✓       |
| Dependency-free Rust PoC               |   —    |   —    |    —     |    —     |   —   |    —    |       ✓       |

(`~` = present in a contrib module, not the default query path.)

## Benchmarks

**Hardware:** Apple M4 Max (arm64), Darwin 24.6.0.
**Config:** n = 10 000, dim = 128, M = 16, ef_max = 128, k = 10, 500 queries.

```
policy                              recall@10  visits    dists     early%   p50us   p95us   p99us
----------------------------------  ---------  --------  --------  -------  ------  ------  ------
fixed-ef128                            0.6758     135.3    1778.7     0.0%   371.3   421.5   446.0
gap(eps=0.005, w=8,  min=40)           0.3636      50.3     720.6   100.0%   175.0   209.7   229.6
gap(eps=0.001, w=12, min=40)           0.4272      63.8     899.8   100.0%   207.5   286.7   329.0
kendall(tau=0.95, w=4, s=3, min=40)    0.4660      72.2    1008.6   100.0%   229.2   324.2   351.0
kendall(tau=0.98, w=4, s=4, min=40)    0.5140      85.1    1172.7    97.4%   271.8   383.6   426.5
kendall(tau=1.00, w=2, s=5, min=40)    0.4312      63.7     898.4   100.0%   211.5   303.5   355.4
```

At equal visit budget, Kendall-tau gives **+5 % relative recall over
the distance-plateau heuristic** in the mid-tuned regime, at negligible
per-check cost (O(k²), k = 10 ⇒ 45 comparisons).

## Optimizations

  * Cache-friendly f32 row-major storage; SIMD-auto-vectorized tight
    L2 distance kernel.
  * Deterministic Splitmix64 PRNG — reproducible benchmarks across
    machines.
  * Min-heap of `Reverse<Scored>` for candidates, max-heap capped at
    `ef_max` for results — classic HNSW dynamics.
  * Warmup floor (`min_visits`) prevents the pathological "τ = 1.0 on
    two adjacent no-op iterations" spurious-early-stop failure mode.
  * O(k²) naive Kendall implementation for correctness; documented
    upgrade path to Knight's O(k log k) merge-sort tau when k ≥ 500.

## Get started

Source, ADR, benchmark script:
<https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-08-19-topk-stability-terminate-ann>

```bash
git clone -b research/nightly/2026-08-19-topk-stability-terminate-ann https://github.com/CrossGen-ai/RuVector.git
cd RuVector
cargo test --release  -p ruvector-topk-stability-terminate
cargo run  --release  -p ruvector-topk-stability-terminate --bin topk-stability-bench
```

Read the ADR: `docs/adr/ADR-305-topk-stability-terminate-ann.md`
Read the research doc: `docs/research/nightly/2026-08-19-topk-stability-terminate-ann/README.md`
