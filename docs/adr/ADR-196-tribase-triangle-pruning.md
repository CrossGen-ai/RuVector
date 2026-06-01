# ADR-196: Triangle-Inequality Pruning for Graph ANN (Tribase)

* **Status**: Proposed (nightly research PoC; not yet integrated into production indexes)
* **Date**: 2026-06-01
* **Owner**: ruvector nightly research
* **Crate**: `crates/ruvector-tribase`

## Context

Graph-based ANN indexes (HNSW, NSG, Vamana/DiskANN) spend ≥80% of query time on f32 distance computations against neighbors expanded during beam search. Several recent systems shrink that cost via *lossy* compression — RaBitQ (1-bit), FINGER (LSH residuals), Anisotropic PQ (learned codebooks). All of them trade some recall for speed.

Tribase (Lu et al., SIGMOD 2024) takes a different path: it stores precomputed landmark distances per node and uses the triangle inequality to derive a **free, exact lower bound** on the query–node distance. Candidates whose lower bound already exceeds the ef-th cutoff can be skipped *without ever changing recall*.

ruvector currently has no lossless distance-pruning layer. Every graph crate (`ruvector-graph`, `ruvector-diskann`, `ruvector-roargraph`) goes straight from "candidate visited" to "full f32 distance." This ADR proposes adopting Tribase as a swappable layer beneath those searchers.

## Decision

Land `crates/ruvector-tribase` as a research-tier crate exposing:

1. A `trait AnnSearcher` with a uniform `search(query, k, ef, &mut stats)` signature.
2. Three reference implementations: `BaselineSearcher` (no pruning), `TribaseSearcher` (single medoid landmark), `MultiLandmarkSearcher` (k random landmarks).
3. A `FlatGraph` substrate so the three variants are measured on identical graphs.
4. A deterministic `tribase_bench` binary that prints real qps / fullD-per-query / pruned-per-query / recall numbers across 4 configs (1k×32, 1k×32 ef=64, 2k×64, 4k×128).
5. Unit tests that assert pruning is lossless (`tribase_preserves_recall`) and monotone in landmark count (`multi_landmark_prunes_more_than_one`).

This is intentionally **not yet integrated** into `ruvector-graph` or `ruvector-diskann`. The PoC is the substrate from which to decide whether to feature-gate Tribase pruning behind a `--features tribase` flag on those crates in a follow-up.

## Consequences

### Positive

* Provably lossless speedup — same recall at strictly fewer full distance computations.
* Tiny per-vector memory cost (4–32 B for k=1–8 landmarks) — sub-7% overhead on a 128-dim store.
* Composes cleanly with adaptive `efSearch`, range-filtered ANN (`ruvector-roargraph`), and Matryoshka multi-resolution retrieval.
* Pruning logic is a single closure passed into `beam_search`, monomorphized per searcher; no v-table cost.

### Negative

* Pruning effectiveness collapses at high dimensions (~3% at D=128 in our PoC) because of concentration-of-distances. At in-RAM, low-D workloads the bound-check overhead can outweigh savings.
* Updates require maintaining `d(x, L)` per landmark; landmark replacement is O(N).
* For inner-product / non-metric spaces, the triangle inequality only holds approximately.

### Neutral / Open

* Landmark selection strategy (medoid vs k-medoids vs learned) is left to follow-up work.
* GPU / SIMD batched pruning is a clear next step but out of scope for this PoC.

## Alternatives considered

* **FINGER-style residual projection** — already explored in nightly 2026-05-10. Lossy, more complex, but tighter at high D.
* **RaBitQ as lower-bound** — explored in nightly 2026-04-23. Lossy with bound; could be combined with Tribase (use the tighter of the two), but stacking them is a separate experiment.
* **Do nothing** — leaves a clean lossless win on the table. Rejected.

## Acceptance

* `cargo build --release -p ruvector-tribase` — green.
* `cargo test -p ruvector-tribase` — 3/3 passing (baseline-finds-self, lossless-recall, monotone-pruning).
* `cargo run -p ruvector-tribase --release --bin tribase_bench` — emits real numbers; recall identical across variants per config.

## Follow-up

* ADR-19X (TBD): feature-gate `tribase` on `ruvector-graph` + `ruvector-diskann`.
* ADR-19X (TBD): learned landmark selection (k-medoids on a sample) and per-cluster landmark routing.
