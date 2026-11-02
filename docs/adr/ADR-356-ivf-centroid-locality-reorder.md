# ADR-356 — IVF Centroid Locality Reorder

- **Status:** Proposed (experimental PoC, 2026-11-02)
- **Owner:** nightly research
- **Related:** ADR-355 (asymmetric-rabitq-q2d1); `ruvector-hnsw-freq-reorder`,
  `ruvector-ef-ivf-posting`.

## Context

IVF indexes in ruvector store a flat per-cluster posting list laid out in
cluster-id order. At query time the engine ranks the `nprobe` nearest
centroids, then scans each cluster's posting list sequentially. The
cluster-id axis today has no spatial meaning: id 7 and id 8 may be far apart
in `R^d` because they came out of k-means in arbitrary init order. Each
hop between consecutive probed clusters is therefore a random jump in the
posting-list arena.

A one-shot permutation of cluster ids — chosen so that id-adjacent clusters
are also space-adjacent — would let the probe-id list, after sorting,
traverse contiguous arenas rather than scattered ones. This is cheap, lossless
(a pure permutation), and composable with every other IVF optimization in
the tree (PQ, ADC fast-scan, PDX, scalar quantization).

## Decision

Introduce `ruvector-ivf-centroid-locality-reorder` as a standalone crate
providing:

- `FlatIvf` — the baseline IVF container.
- `Reorder` trait — pluggable permutation strategy.
- `GreedyChain`, `ProjectionSort` — two reference strategies.
- `apply_permutation` — rewrite the physical layout.

Keep it crate-local, feature-gated, and off by default on the production
IVF path until (a) real-dataset benches confirm the win and (b) the chain
is extended with k-means++ init / 2-opt refinement.

## Consequences

**Positive.**
- Mechanism verified end-to-end on synthetic data (property test enforces
  recall equality; stride proxy drops 0.6–0.9 %).
- Zero runtime overhead: permutation is build-time only; query path only
  gains an `sort_unstable` on `nprobe` ids.
- Composes with any other IVF trick (PQ ADC, variance-adaptive SQ, PDX,
  cascade ADC) because it changes only the id-to-arena mapping.
- Trait design allows swapping in Hilbert-order, TSP, or ML-based
  reorderings without touching the index.

**Negative / Risk.**
- Measured speedup on dim=128, 512 clusters is ~1.01×. The win is small
  in the regime where posting lists are already larger than L1. See
  "Practical failure modes" in the research doc.
- Permutation is one-shot: streaming inserts/deletes that add centroids
  invalidate the mapping. A production variant needs an online strategy
  (e.g. insert at the id that minimizes local chain stretch).
- The stride proxy is not a cycle-counter metric. Confirming the mechanism
  on a Linux box with `perf stat -e L1-dcache-load-misses` is a hard
  prerequisite before claiming this is a cache win rather than a
  prefetch-friendliness nudge.

## Alternatives considered

1. **Do nothing.** Leave cluster ids in k-means order. Baseline today;
   loses the free locality.
2. **Reverse Cuthill-McKee on the probe-cooccurrence graph.** Treat each
   query batch as a graph of cluster co-visits and reorder to minimize
   bandwidth. Needs a representative query workload at build time, which
   many deployments don't have.
3. **Hilbert / Z-order on multi-axis random projections.** Captures more
   variance than a single-axis projection but introduces hyperparameters
   (number of axes). Logged as roadmap item.
4. **Full TSP (Christofides, 2-opt).** Tighter chain, same `O(n_c² · d)`
   up front, plus the optimization pass. Deferred until the regime where
   greedy chain shows a bigger win.

## Follow-ups

- Implement roadmap items 1–3 from the research doc.
- Add a `criterion.rs` harness and a feature-gated SIFT1M regression test.
- Measure real cache miss rates on a Linux box; if the mechanism is
  confirmed, propose exposing `enable_centroid_reorder: bool` on the
  ruvector IVF builder.
