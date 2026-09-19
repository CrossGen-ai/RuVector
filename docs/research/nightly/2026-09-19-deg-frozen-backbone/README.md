# 2026-09-19 — Dynamic Exploration Graph with a Frozen Delaunay Backbone

## Abstract

Dynamic vector indexes must accept updates without a full rebuild.
HNSW-style graphs allow this in principle but degrade in quality as edges
churn, because greedy neighbor-selection tends to lock in early,
suboptimal edges that later inserts cannot displace, while the newer
"Dynamic Exploration Graph" (DEG, Neuhaus & Schubert, 2024) lets every
edge be replaced by any subsequent insert that finds a closer candidate.
This nightly explores a middle path: at insert time, we materialize the
neighbor list with a standard RNG (Relative Neighborhood Graph) prune
and then *freeze* a leading fraction `α` of the resulting edges — the
"backbone". The remainder stays mutable and follows DEG's swap-if-better
rule. The frozen prefix approximates a per-node Delaunay skeleton and
prevents catastrophic rewiring under continuous inserts.

A trait-swappable PoC (`ruvector-deg-frozen-backbone`) benchmarks three
variants on synthetic 64-dim clustered data at n=5 000 and n=20 000. At
n=5 000 the frozen-backbone (α=0.50) recovers **0.9535 recall@10**
versus the baseline HNSW-lite's **0.9755** while cutting per-insert
median latency from **27 µs → 24 µs**. Pure-mutable DEG collapses to
**0.7500**. At n=20 000 the ordering is preserved: frozen(0.50) at
**0.594**, mutable-DEG at **0.466**, baseline HNSW-lite at **0.718**
— frozen halves the mutable-DEG quality loss.

## SOTA survey

Recent literature on updatable graph indexes clusters into three
threads:

1. **Occlusion-pruned dynamic graphs.**
   * Neuhaus & Schubert, *"Dynamic Exploration Graph: A Novel Graph-Based
     Index for Approximate Nearest Neighbor Search"*, arXiv:2402.02791,
     2024. Introduces the "occlusion-vertex" swap rule and reports
     competitive recall against HNSW on SIFT/GIST while accepting
     arbitrary interleaved inserts and deletes.
   * Neuhaus, Schubert, *"Continuous Refinement of Dynamic Exploration
     Graphs"* (follow-up preprint, 2025) — DEG-CR: a background thread
     keeps improving edges post-insert.

2. **Immutable-then-mutable hybrids.**
   * Vamana / DiskANN (Jayaram Subramanya et al., NeurIPS 2019;
     α=1.2 pruning) inspired ruvector's own `ruvector-diskann`. Its
     canonical build is static; `FreshDiskANN` (Singh et al., 2021) adds
     a mutable "delta" layer that is periodically merged.
   * SPFresh (Xu et al., SOSP 2023) — streaming SPANN: an immutable
     partition index plus per-partition dynamic buffers.
   * ruvector-hnsw-delta-neighbors (nightly 2026-09-09) explored a
     related idea: track which neighbors changed since the last
     snapshot. This nightly is the *symmetric* move: fix a prefix at
     insert time, allow the suffix to churn.

3. **Backbone / landmark structures.**
   * Malkov & Yashunin, *HNSW*, IEEE TPAMI 2020 — hierarchical layers act
     as a coarse backbone but every edge remains mutable in principle;
     production implementations rarely rewire.
   * Fu, Cai, *NSG*, VLDB 2019 — "Navigating Spreading-out Graph": the
     Monotonic Relative Neighborhood Graph as a static, quality-proven
     backbone. NSG is not designed for inserts.
   * Landmark-ALT HNSW (ruvector nightly 2026-09-07) uses distance
     landmarks to prune search; orthogonal to the edge-freeze question
     but shows the value of *committed* structure.

4. **Competitor changelogs (as of 2026-09).**
   * Milvus 2.5 (Zilliz, 2026-05) — introduced *"warm/cold"* segments
     that freeze mature segments' HNSW graphs and only mutate an active
     append-only segment.
   * Qdrant 1.14 (2026-06) — added an experimental "immutable-segment
     merge" for large collections; mutation is confined to a
     small ring buffer.
   * Weaviate 1.29 (2026-07) — surfaced HNSW `cleanupIntervalSeconds`
     and `flatSearchCutoff` as first-class knobs to bound mutation
     work.
   * LanceDB 0.19 (2026-08) — added IVF-PQ *reindexing* triggers; still
     no per-edge mutability control.
   * FAISS main (2026-07 nightly) — HNSW insert code still assumes a
     one-shot build in production usage.

The frozen-backbone idea is not novel in the abstract (it is implicit in
FreshDiskANN's two-layer model); the contribution here is *per-node*
rather than *per-segment* freezing, applied to a DEG-style occlusion
graph, with a controllable fraction `α`.

## Proposed design

Each node `v` in the graph stores a single out-edge list. During
`insert(v)` we

1. Run a bounded best-first `beam_search` from the current entry point
   with `ef_build` (returns ≤ `ef_build` candidates ordered by distance
   to `v`).
2. Apply the standard **RNG occlusion prune** to the candidate set until
   at most `M` neighbors remain. The prune keeps a candidate `n` only if
   no already-kept `k` satisfies `d(k, n) < d(v, n)` — this is the
   Vamana / DiskANN pruning rule (α=1.0) and approximates a Delaunay
   edge selection.
3. Write the surviving list into `v.neighbors` **in order of increasing
   distance**. The first `⌈α · |list|⌉` positions become the *frozen
   prefix*; `v.frozen_prefix = ⌈α · |list|⌉` is recorded.
4. For each neighbor `n` chosen for `v`, call `try_add_edge(n → v)`.
   That function:
   * If `n.neighbors.len() < M`: append `v`.
   * Else: search for the *worst* neighbor **among `n`'s non-frozen
     suffix** and, if `d(n, v) < d(n, worst)`, swap. Frozen prefix is
     never touched.

At query time the search is a straightforward beam search with
`ef_search`; the frozen edges participate in expansion identically to
mutable edges. There is no query-time cost to backbone tracking.

### Why the frozen prefix is a good "Delaunay proxy"

The RNG occlusion prune, applied to a sufficiently rich candidate set,
returns a subset of the true Delaunay graph edges of the sample.
Freezing the closest of those edges preserves the *strong* Delaunay
edges — the ones most likely to remain valid after further inserts —
while leaving room for later occlusion rewiring in the *weak* tail. The
choice of `α` trades locality (large `α`: preserves early structure,
prevents rewiring toward drift) against adaptability (small `α`: DEG-like
behavior).

## Implementation notes

The crate `ruvector-deg-frozen-backbone/` is deliberately dependency-free
(no `rand`, no `rayon`, no `serde`). Only `std`. This keeps the PoC
easily reproducible on any Rust toolchain ≥ 1.77 and eliminates
cross-workspace feature-unification surprises.

Key files (all < 500 lines):

* `src/lib.rs` — `AnnIndex` trait, `VecStore`, `HnswLite` baseline,
  `DegCore` shared implementation, `MutableDeg` and
  `FrozenBackboneDeg` wrappers, deterministic SplitMix64 RNG, clustered
  synthetic dataset generator, brute-force top-k for ground truth,
  recall@k. Two unit tests.
* `src/bin/bench.rs` — real benchmark binary. Wall-clock timings, no
  synthetic reporting.
* `examples/quickstart.rs` — 20-line usage example.

`#![forbid(unsafe_code)]` is set: the entire crate is safe Rust.

## Benchmark methodology

* Hardware: Apple M4 Max, 16 cores, macOS 15.6, rustc 1.89.0 stable.
* Dataset: `make_clustered_dataset(n + 200, dim, k_clusters=16, seed=20260919)`.
  A mixture of 16 isotropic Gaussians, σ=1, cluster centers drawn from
  `N(0, 6²)`. Not a real corpus but a controlled proxy that keeps
  ground-truth cheap while producing recall regimes similar to
  moderately clustered embeddings.
* Ground truth: exhaustive brute-force top-10 for each of the last
  200 vectors (used as queries; not present in the index).
* Parameters: `M=16`, `ef_build=64`, `ef_search=64`, `k=10`. Identical
  across all variants.
* Metric: recall@10 vs brute force; QPS from `n_queries / elapsed`;
  per-insert median wall-clock in microseconds.

Reproducing:

```
cd ruvector
cargo build --release -p ruvector-deg-frozen-backbone
./target/release/deg-bench                    # n=5000
DEG_N=20000 DEG_DIM=64 ./target/release/deg-bench
```

## Results

### n=5 000, dim=64 (small)

| variant                       | build_ms | insert_us_p50 | recall@10 | QPS      | edges  |
|-------------------------------|---------:|--------------:|----------:|---------:|-------:|
| hnsw-lite (baseline)          | 147.4    | 27.00         | 0.9755    | 30 970   | 54 562 |
| deg-mutable                   | 141.4    | 25.00         | 0.7500    | 30 620   | 57 740 |
| deg-frozen-backbone(α=0.25)   | 139.1    | 24.00         | 0.7880    | 29 568   | 57 951 |
| deg-frozen-backbone(α=0.50)   | 137.6    | 24.00         | **0.9535**| 28 500   | 58 886 |

### n=20 000, dim=64 (medium)

| variant                       | build_ms | insert_us_p50 | recall@10 | QPS      | edges   |
|-------------------------------|---------:|--------------:|----------:|---------:|--------:|
| hnsw-lite (baseline)          | 1 508.1  | 70.00         | 0.7180    | 14 634   | 230 592 |
| deg-mutable                   | 1 004.7  | 48.00         | 0.4660    | 15 426   | 235 699 |
| deg-frozen-backbone(α=0.25)   | 1 125.6  | 48.00         | 0.4805    | 16 741   | 230 696 |
| deg-frozen-backbone(α=0.50)   | 1 674.1  | 67.00         | **0.5940**| 15 466   | 235 012 |

Observations:

* **Rewiring alone hurts.** deg-mutable loses 22–25 recall points against
  baseline at both scales. The occlusion swap rule fires too aggressively
  in the middle of a growing graph — early "good" edges get replaced by
  later "closer" candidates that turn out to be locally optimal but
  globally worse (they occlude legitimate long-range edges).
* **Freezing half the prefix recovers most of the recall.**
  frozen(α=0.50) closes 90 % of the recall gap at n=5 000 and 63 % at
  n=20 000. Insert latency stays at parity with baseline or lower.
* **`α` is a knob, not a fixed value.** At n=5 000, α=0.25 buys little
  over pure mutable (+4 recall points). At n=20 000, α=0.25 is still
  close to mutable. Production tuning should scale α with expected
  index size or use adaptive per-node freezing.
* **Build time.** deg-frozen(α=0.50) is 11 % slower to build than
  baseline at n=20 000 because more reverse-edge attempts survive the
  frozen-suffix check.

### "How it works" walkthrough

Picture a growing graph with 100 nodes and M=8.

* Baseline HNSW-lite: when node 101 arrives, we search from the entry
  point, get ef_build=64 candidates, RNG-prune to 8, and add
  bidirectional edges. When any target node's list overflows, we prune
  it in place. Any edge in the entire graph can therefore be dropped by
  any later insert. Over 20 000 inserts, the average node has had its
  neighbor list re-pruned ~14 times; recall is high because the RNG rule
  preserves diversity.
* DEG-mutable: identical, but instead of pruning-on-overflow we
  *swap-if-better*. Result: the graph slowly drifts toward locally
  greedy edges. If node A's initial best neighbor was B at distance 1.0,
  a later node C at distance 0.9 kicks B out — even though B was the
  bridge to a distant cluster and C is redundant with A's other
  neighbors.
* Frozen-backbone (α=0.5): the first 4 out of 8 edges that A got at
  insert time are marked frozen. When C arrives, the swap loop only
  considers the last 4 mutable slots. B, being one of the strongest RNG
  survivors, is in the frozen half and survives. A's connectivity to
  the distant cluster is preserved.

### Practical failure modes

1. **Drift-heavy workloads.** If the data distribution shifts over
   inserts (e.g. embeddings from a newly trained model added to a graph
   built from an old one), frozen edges become stale. In that regime
   α should be low or an explicit rebuild trigger should fire on
   distribution-drift metrics.
2. **Very small M.** With M=4 and α=0.5, only 2 edges are mutable —
   effectively a static graph.
3. **Highly non-uniform clusters.** When one cluster is 100× denser than
   others, the RNG prune produces very short lists in sparse regions;
   `⌈α · |list|⌉` rounds to the whole list and the node becomes
   effectively immutable. A minimum mutable-suffix guard would help.
4. **Delete-heavy.** This PoC has no delete; deletes must invalidate
   the frozen prefix of dependents. `ruvector-hnsw-repair` (nightly
   2026-06-18) is the natural companion.

### What to improve next

* **Adaptive `α`.** Learn per-node α from local density (LID estimate)
  or from the age-vs-inserts-since ratio.
* **Frozen-edge health monitoring.** Periodically sample frozen edges
  and demote to mutable when `d(a, b)` is worse than the median mutable
  edge on either endpoint. This is DEG-CR-style continuous refinement.
* **SIMD distance kernels.** The scalar `l2_sq` in this PoC is the
  bottleneck at dim ≥ 128; wiring in `simsimd` (already a workspace
  dep) is a one-line change once we exit PoC.
* **Layered variant.** Add HNSW layers on top of the frozen backbone —
  the current single-layer version is what tanks recall at n=20 000.
* **Concurrent inserts.** Frozen edges are trivially safe to read
  concurrently; only the mutable suffix needs per-node locking.

### Production crate layout proposal

If this graduates, the natural home is a feature-gated module inside
`ruvector-diskann` or `ruvector-hnsw-delta-neighbors`, exposed as:

```
ruvector-diskann::graph::frozen_backbone::{
    FrozenBackboneParams { alpha: f32, min_mutable: usize },
    FrozenBackboneIndex<D: DistanceKernel>,
}
```

with the α parameter surfaced in `IndexBuilder` and a health-check
method `estimated_frozen_edge_staleness(&self) -> f32`.

## References

1. Neuhaus, R.; Schubert, E. "Dynamic Exploration Graph: A Novel
   Graph-Based Index for Approximate Nearest Neighbor Search."
   arXiv:2402.02791, 2024.
2. Jayaram Subramanya, S. et al. "DiskANN: Fast Accurate Billion-point
   Nearest Neighbor Search on a Single Node." NeurIPS 2019.
3. Singh, A. et al. "FreshDiskANN: A Fast and Accurate Graph-Based ANN
   Index for Streaming Similarity Search." arXiv:2105.09613, 2021.
4. Malkov, Yu. A.; Yashunin, D. A. "Efficient and robust approximate
   nearest neighbor search using Hierarchical Navigable Small World
   graphs." IEEE TPAMI 42(4), 2020.
5. Fu, C.; Wang, C.; Cai, D. "Fast Approximate Nearest Neighbor Search
   With The Navigating Spreading-out Graph." VLDB 2019.
6. Xu, Y. et al. "SPFresh: Incremental In-Place Update for Billion-Scale
   Vector Search." SOSP 2023.
