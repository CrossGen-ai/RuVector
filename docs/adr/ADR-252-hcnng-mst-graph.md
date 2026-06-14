# ADR-252: HCNNG (Hierarchical Clustering Navigating Neighbor Graph)

**Status:** Accepted (PoC) · 2026-06-14
**Branch:** `research/nightly/2026-06-14-hcnng-mst-graph`
**Crate:** `crates/ruvector-hcnng`

## Context

ruvector already ships HNSW (via `hnsw_rs`), DiskANN (`ruvector-diskann`),
DEG, NSG/MRNG explorations, ROARgraph, and several IVF variants. Each
single-shot graph builder has strong recall but at least one of these
costs:

- HNSW requires a globally ordered insertion stream to maintain layer
  invariants; concurrent build is non-trivial.
- NSG/MRNG must build a full kNN graph upfront (typically via NN-Descent),
  which is O(n^1.14)–O(n^1.4) and dominates wall-clock.
- DiskANN tunes the alpha-RNG pruning rule per dataset.

We lack a **fully embarrassingly-parallel, parameter-light** graph builder
that can grow incrementally (one more tree → one more set of edges →
union into the existing graph). HCNNG (Munoz et al., Pattern
Recognition 2019) fits exactly this slot.

## Decision

Add `ruvector-hcnng` as a new workspace crate. It implements:

- Recursive 2-pivot metric partition trees → leaves of bounded size.
- Prim's MST per leaf, optionally augmented with `knn_per_node` extra
  intra-leaf kNN edges.
- Multi-entry beam search (top-K hubs + deterministic random anchors).
- Trait-based `Distance` (L2², neg-IP, cosine) for backend swappability.
- A demo benchmark binary that reports brute-force vs HCNNG with three
  parameter regimes and asserts a recall floor on the best variant.

Initial defaults: `n_trees=10`, `leaf_size=32`, `max_degree=32`,
`knn_per_node=3`, `ef_search=64`. These match the paper's regime for
the SIFT1M-class workloads and give the user a working knob set without
tuning.

## Consequences

### Positive

- New builder option with O(n · log n · n_trees) construction, fully
  parallelizable across trees (next PR).
- No insertion order dependency; adding a tree is composable.
- Tiny dependency surface — only `rand`, `thiserror`, `serde`. No
  hnsw_rs / simsimd reach-in.
- Files are under 500 lines each; clean trait boundaries leave room for
  RaBitQ/LVQ distance backends, NAPI/WASM wrappers, and rkyv snapshots.

### Negative

- PoC build is single-threaded — slower than HNSW above ~100k vectors
  until rayon-ization (planned).
- Recall on highly clustered data (32 well-separated Gaussians, σ=0.6)
  drops to 0.3–0.5; mitigations are documented in the research note but
  not in this initial cut.
- Each tree pays an L²-distance-matrix cost per leaf when
  `knn_per_node > 0`. Cap leaf_size accordingly.

### Risks

- Adopting HCNNG as a default could regress users who tuned HNSW knobs
  for their data — keep it as a separate index type, not a swap-in.

## Alternatives considered

1. **Reuse hnsw_rs with multiple entry points only.** Doesn't address
   the parallel-build or incremental-tree-union ergonomic. Rejected.

2. **Extend `ruvector-diskann` with alpha-RNG only.** DiskANN's strength
   is SSD-tuned single-graph, not parallel ensembles. Different niche.
   Keep both.

3. **Skip HCNNG; build NN-Descent + NSG instead.** NSG has slightly
   higher recall ceiling, but NN-Descent dominates build cost and is
   already on the roadmap (`research/nightly/2026-05-27-nn-descent-graph-build`).
   HCNNG complements rather than replaces it.

4. **Use a single random projection forest, no MST.** This is the
   `n_trees=1` ablation — recall@10 ≈ 0.014 at d=64. Rejected.

## References

- Munoz, Gonzalez, Buhmann. *Hierarchical Clustering-Based Graphs for
  Large Scale Approximate Nearest Neighbor Search.* Pattern Recognition
  96 (2019), 106985.
- Research note: `docs/research/nightly/2026-06-14-hcnng-mst-graph/README.md`
- ADR-178 (graph-family integration plan).
- ADR-251 (most recent ADR; this is the next number).
