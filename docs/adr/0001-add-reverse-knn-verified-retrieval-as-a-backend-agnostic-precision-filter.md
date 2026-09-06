<!-- One decision, stated in the filename (ADR-0021). -->

# ADR-0001: Add Reverse-KNN Verified Retrieval as a backend-agnostic precision filter

> Decision date: 2026-09-06
> Status: Proposed
> Scope: ruvector query pipeline (post-ANN, pre-rerank)
> Drivers: hub contamination in agent-memory retrieval; missing composable precision filter that works across HNSW/IVF-PQ/RaBitQ/DiskANN

## Context

ruvector ships many candidate-generating ANN backends (HNSW, IVF-PQ, RaBitQ,
DiskANN, LSM-ANN, cascade-ADC, …) and several precision-improvement primitives
(fused RaBitQ residual, GNN reranker, semantic query cache). What it does not
have is a *cheap, backend-agnostic post-hoc precision filter* that sits between
a candidate-generating pass and the final rerank/return step.

The forcing failure mode is asymmetric neighborhoods — the classical *hubness*
phenomenon in high-dimensional ANN — where a small subset of dataset points
appear near many queries but whose own tight neighborhood is dominated by
other hubs. In agent-memory workloads this manifests as "generic" memories
with high embedding norm distracting retrieval across every unrelated query.
State-of-the-art responses either (a) mutate the graph (SymphonyQG,
RoarGraph — couples the fix to one backend), (b) rescale distances (Mutual
Proximity — requires reindexing), or (c) LLM-rerank (very expensive). None
compose cleanly with our existing multi-backend stack.

## Decision

1. Ship a new leaf crate `crates/ruvector-rknn-verified/` implementing a
   backend-agnostic Reverse-KNN Verified Retrieval filter with two modes
   (`Live`, `Cached`) parameterised by `(k_rev, slack)`.
2. Define an `NnIndex` trait as the sole coupling point so any existing
   backend can be plugged in via a one-function adapter.
3. Include a deterministic synthetic benchmark that measures recall,
   precision, hub-in-result rate, and per-query latency, and exits non-zero
   on precision or latency regressions (numeric acceptance test).
4. Take zero external dependencies (pure `std`) in the initial crate;
   adapter crates or `serde` feature-flags come later.

## Alternatives Considered

* **Mutual Proximity distance rescaling** (Schnitzer et al. 2012). More
  invasive; changes the distance function itself and requires
  reindexing/retraining. Rejected as the first step.
* **Graph-side edge symmetrization** (SymphonyQG). Couples the fix to
  HNSW. Rejected: we want a filter that composes with every backend.
* **LLM reranker.** Complementary but ~1000× the latency of an RkNN cache
  lookup. RkNN reduces the candidate set an LLM must judge; not a
  substitute.
* **Do nothing / brute-force distance recompute.** The current state. Loses
  both the precision gain and the asymmetric-neighborhood signal.

## Consequences

Positive:

* +10.4 pp precision@10 on the controlled hub-contaminated benchmark
  (n=6000, dim=64, hub_frac=6%, hub_scale=3.8×), cached mode.
* ~5% latency overhead in cached mode over the baseline noisy-ANN pass
  (105 µs → 110 µs median per query, single-thread Apple Silicon).
* Cache memory ~100 bytes/point at `k_rev = 12`.
* Backend-agnostic; composes with HNSW/IVF-PQ/RaBitQ/DiskANN via one
  `NnIndex` impl each.
* Zero new workspace dependencies.

Negative / risks:

* Cache is a snapshot; streaming inserts need a rebuild schedule (future
  work).
* `slack` must be tuned per-distribution — bad values can starve the
  return set. Mitigation documented: fall back to top-K if
  `|filtered| < K/2`.
* Adds per-query O(M · dim) work in cached mode — acceptable at M ≤ 64.

## Testable Criteria

| ID | Criterion | How verified |
|----|-----------|--------------|
| TC-1 | cached-mode precision@10 ≥ baseline precision@10 on the shipped synthetic benchmark | `cargo run --release -p ruvector-rknn-verified --bin benchmark` prints acceptance line "cached_precision >= baseline_precision : true" |
| TC-2 | cached-mode median per-query latency ≤ 4× live-mode median on the same benchmark | Same run prints "cached_latency <= 4x live_latency : true" |
| TC-3 | crate builds and all unit tests pass with zero external dependencies | `cargo build --release -p ruvector-rknn-verified` and `cargo test --release -p ruvector-rknn-verified` both exit 0; `Cargo.toml` has no `[dependencies]` section |

## References

* Radovanović, Nanopoulos, Ivanović. *Hubs in Space: Popular Nearest
  Neighbors in High-Dimensional Data.* JMLR 11 (2010).
* Korn, Muthukrishnan. *Influence Sets Based on Reverse Nearest Neighbor
  Queries.* SIGMOD 2000.
* Schnitzer, Flexer, Schedl, Widmer. *Local and Global Scaling Reduce
  Hubs in Space.* JMLR 13 (2012).
* Chen et al. *RoarGraph.* VLDB 2024.
* Gao et al. *SymphonyQG.* SIGMOD 2025.
* Research doc: `docs/research/nightly/2026-09-06-rknn-verified-retrieval/README.md`
