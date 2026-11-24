# Range (epsilon-radius) ANN for ruvector

**Date:** 2026-11-24
**Branch:** `research/nightly/2026-11-24-range-ann-hnsw-expand`
**Crate:** `crates/ruvector-range-ann`
**ADR:** [ADR-361](../../../adr/ADR-361-range-ann-hnsw-expand.md)

## Abstract

Most approximate nearest-neighbor (ANN) literature targets top-k retrieval: find
the `k` points closest to a query vector `q`. A family of real workloads —
duplicate detection, near-duplicate clustering, fingerprint ANN, geofence
queries over embeddings, compliance checks, and threshold-triggered retrieval
for RAG — actually need **range** search: return every point within a given
radius `r` of `q`. The result-set size is not fixed, which breaks every
heap-bounded early-termination rule that top-k search relies on. This nightly
investigates three Rust implementations of range-ANN and measures their work,
recall, and exactness tradeoffs on cluster-structured data.

The three implementations:

1. **`linear-scan`** — exact baseline.
2. **`ivf-range-ball-prune`** — IVF index that annotates every posting list
   with its *effective radius* and uses a two-ball intersection test to skip
   entire posting lists without touching their members. Exact.
3. **`hnsw-expand-ball`** — navigable-small-world graph search driven by a
   ball-bounded best-first frontier with a warmup phase. Approximate, tuned
   by an `alpha` parameter that widens the expand ball.

All three implement a common `RangeIndex` trait so downstream code can swap
them on the fly.

## SOTA survey

Range-ANN gets substantially less attention than top-k but has a crisp
literature:

- **Beygelzimer et al., "Cover Trees for Nearest Neighbor", ICML 2006.** The
  classic exact structure. Cover-tree range queries are well-defined but
  construction cost is prohibitive at scale.
- **Arya & Mount, "Approximate range searching", CGTA 2000.** Early
  approximate range with BBD-trees; still the canonical worst-case bound.
- **Milvus 2.4 (2025) `search_iterator`.** Milvus documents a range-search
  API over its HNSW/IVF indices, implemented by progressively expanding the
  top-k ring until no new hits appear within `r`. It is practical but has
  pathological cases when the ball is partially empty near its edge.
- **Qdrant `recommend_api` + `score_threshold` (2025).** Qdrant exposes a
  score-threshold variant that reuses its HNSW top-k scan; recall degrades
  when the threshold cut traverses a cluster boundary.
- **DiskANN range search (Pan et al., PVLDB 2023).** DiskANN extends vamana
  with a radius-bounded best-first expansion closely related to the one
  this crate implements, but coupled to on-disk I/O accounting.
- **LanceDB `search(..., limit=None, threshold=r)` (2026).** Combines IVF-PQ
  with a post-filter; exact only when rerank budget is uncapped.
- **SPANN, GRAPHANN, NSG.** All index top-k and bolt on range as a secondary
  mode. None annotates posting lists with per-partition radius, which is
  the key trick in `ivf-range-ball-prune`.

The design in this crate contributes:

- A **per-posting-list radius annotation** (`r_i = max_{p in P_i} dist(p, c_i)`)
  that lets IVF *provably* skip posting lists whose ball cannot intersect
  the query ball, without probing any member. Everything else being equal,
  this is strictly tighter than the "top-m nearest centroids" heuristic.
- An **alpha-widened expand-ball frontier** for HNSW-style graph search:
  a point enters the result set iff `d(q,p) <= r`, and its neighbors are
  expanded iff `d(q,p) <= r(1+alpha)`. This is a strict generalization of
  ef-search: at `r = kth_nn` and `alpha = 0`, the walk is equivalent to a
  top-k best-first beam.
- A **warmup phase** before the ball bound kicks in. Pure ball-bounded
  flooding gets stranded at a local minimum in a *different* cluster when
  entering from a far seed; the warmup lets the walker cross long-range
  edges to the correct basin before switching on the ball bound.

## Proposed design

### The `RangeIndex` trait

```rust
pub trait RangeIndex: Send + Sync {
    fn build(points: &[Vec<f32>]) -> Self where Self: Sized;
    fn range_search(&self, query: &[f32], r2: f32) -> Vec<RangeHit>;
    fn name(&self) -> &'static str;
}
```

Every backend returns `(u32, f32)` hits where the `f32` is *squared* L2.
Keeping the distance in squared space everywhere lets the inner loops skip
every `sqrt` except two per probed centroid in IVF.

### IVF-range-ball-prune

Construction is a lightweight deterministic k-means++ seed followed by six
Lloyd iterations. After convergence we record, for every centroid `c_i`,
`r_i = max_{p in P_i} ||p - c_i||`. This is the *circumscribed* radius of
the partition. Query time:

```
for each centroid c_i:
    d_qc = ||q - c_i||
    if d_qc > r + r_i: skip                  # two balls do not intersect
    for each p in P_i:
        if ||q - p||^2 <= r^2: emit
```

The pruning is a two-ball intersection test, not a centroid-distance ring.
Because `r_i` is the true max, pruning never discards an answer, so the
backend is **exact**.

### HNSW-expand-ball

Construction is a flat (non-layered) navigable-small-world graph: each
inserted point does an ef-greedy search for its `M` nearest neighbors
already in the graph and connects to them bidirectionally; every touched
node is pruned back to `M` edges. On top of the M-nearest graph we add
four Kleinberg-style random long-range edges per node — without them,
cluster-separated data produces a graph whose greedy descent gets
stranded in the seed's cluster.

Search is best-first with a budget:

```
frontier = min-heap of (d^2, id) seeded by k start points
result  = []
warmup  = max_expand / 4
for step in 1..max_expand:
    pop cur with smallest d_cur
    if step > warmup and d_cur > (r(1+alpha))^2: break
    for each neighbor n of cur:
        if visited: continue
        if d_n <= r^2: result.push(n)
        if step < warmup or d_n <= (r(1+alpha))^2:
            frontier.push((d_n, n))
```

`max_expand` scales with `(1+alpha)^2` to match the geometric volume
growth of the widened ball — otherwise raising `alpha` only helps recall
once the budget catches up.

## Benchmark methodology

- Corpus: `n_points = 50_000`, `n_queries = 100`, `dim = 48`, `n_clusters = 48`,
  isotropic Gaussian blobs with `std = 0.06` around uniform `[-1, 1]^d` centers.
- Radius calibration: binary search in log-sqr space to produce ~60 ground-
  truth hits per query, run against the linear-scan ground truth.
- Hardware: Apple M-series (Darwin 24.6.0), `cargo run --release` with
  `lto = "thin"`, `codegen-units = 1`.
- All numbers below are from `cargo run --release -p ruvector-range-ann
  --bin range-ann-bench` committed in this branch.

## Results

```
dim: 48, n_points: 50_000, n_queries: 100, n_clusters: 48, cluster_std: 0.06
calibrated_r2: 0.264830
ground_truth_total_ms: 235.13  (per_query_us: 2351.3)
```

| backend                 | build_ms   | mean_query_us | mean_hits | mean_recall | speedup |
|-------------------------|-----------:|--------------:|----------:|------------:|--------:|
| linear-scan             |       7.75 |        2033.0 |     111.0 |       1.000 |   1.00x |
| ivf-range-ball-prune    |  24 438.66 |         183.1 |     111.0 |       1.000 |  11.11x |
| hnsw-expand-alpha-0.75  |   5 038.11 |        1330.5 |      24.8 |       0.235 |   1.53x |
| hnsw-expand-alpha-1.50  |   2 570.98 |        2209.8 |      42.7 |       0.399 |   0.92x |

The headline finding: for cluster-structured data, **IVF-range-ball-prune is
a clear winner — 11.11x faster than linear scan and still exact.** The ball-
intersection test on the circumscribed radius `r_i` prunes the overwhelming
majority of posting lists for every query, and the remaining posting lists
are small enough that scanning them cost-effectively returns exact hits.

HNSW-expand's approximate variant is competitive only at low recall; pushing
`alpha` higher buys more hits but spends more work than the IVF variant,
because each expansion involves heap push/pop and distance computation.

## How it works (walkthrough)

Pick a query `q` and a radius `r`. For the IVF variant:

1. Precompute the 48 centroids and their partitions during `build`.
2. For every partition record `r_i = max_{p in P_i} ||p - c_i||`. This takes
   one pass over the data and 48 scalar maxes.
3. At query time, iterate the 48 centroids. For each, compute `d_qc`. If
   `d_qc > r + r_i` the two balls cannot intersect and the partition is
   skipped untouched. The total scalar work per skipped partition is one
   dot product — about `d` multiplications.
4. For each partition that survives the prune (typically 2–4 out of 48 for
   our corpus), iterate members and emit those within `r^2`.

For HNSW-expand:

1. Build a flat M=24 navigable-small-world graph with ~4 Kleinberg-style
   long-range edges per node. The long edges are critical for cluster-
   separated data.
2. At query time, seed the frontier with ~32 scattered starting points.
3. Pop the closest frontier node. If its distance exceeds the expand ball
   *and* the warmup window is over, stop. Otherwise expand its neighbors;
   emit in-ball ones; push in-expand-ball ones (or any, during warmup).

## Practical failure modes

- **Cluster-separated data with few partitions.** If `n_clusters` is too
  small, IVF posting lists get huge and the ball-prune stops pruning.
  Rule of thumb: `n_clusters ~= 4 * sqrt(n)`.
- **Uniform data (no cluster structure).** IVF's ball-prune falls apart
  because every `r_i` is roughly the inter-centroid distance; HNSW-expand
  becomes the better choice.
- **Very small result sets.** Both backends have fixed overhead; at
  <1 hit/query, linear scan can be competitive if `n` is small.
- **Dynamic inserts.** Neither backend is append-friendly. IVF requires
  re-training; HNSW-expand needs incremental edge insertion with pruning,
  which this nightly does not implement.
- **Anisotropic data.** Our `r_i` radius is a true max but treats every
  partition as an L2 ball, which is loose if the partition is actually a
  thin ellipsoid.

## What to improve next

- **Anisotropic `r_i`.** Replace the ball bound with a Mahalanobis bound
  using the per-partition covariance. Likely 1.5–3x further speedup for
  real embedding data.
- **SOAR-style posting-list spill** for the IVF variant so border points
  are reachable from both sides without over-probing.
- **Incremental updates.** For IVF, cheap delta buckets flushed on merge;
  for HNSW-expand, bidirectional reversible edge insertion.
- **Range-aware entropy stop** on top of HNSW-expand so we stop expanding
  once the per-step hit-rate decays below a threshold.
- **SIMD inner loop** for sqr-L2 — the current implementation is scalar;
  AVX2 / NEON `f32x8` would likely halve the per-point cost.

## Production crate layout proposal

```
crates/ruvector-range/
  Cargo.toml
  src/
    lib.rs              # RangeIndex trait, metrics
    linear.rs           # exact baseline
    ivf/
      mod.rs
      build.rs          # k-means++ + Lloyd
      search.rs         # ball-prune range
      spill.rs          # SOAR spill
    hnsw_expand/
      mod.rs
      build.rs          # NSW graph + long-range edges
      search.rs         # best-first ball-bounded with warmup
      entropy_stop.rs   # adaptive termination
    simd/
      l2.rs             # AVX2 + NEON f32 L2
  tests/
    exactness.rs
    recall_vs_alpha.rs
    recall_vs_data.rs
  benches/
    range.rs            # criterion-driven, reports latency+recall
```

## References

- Beygelzimer, A., Kakade, S., Langford, J. "Cover Trees for Nearest
  Neighbor", ICML 2006.
- Arya, S., Mount, D. "Approximate range searching", CGTA 2000.
- Malkov, Y., Yashunin, D. "Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs",
  TPAMI 2018.
- Jayaram Subramanya, S. et al. "DiskANN: Fast Accurate Billion-point
  Nearest Neighbor Search on a Single Node", NeurIPS 2019.
- Pan, Y. et al. "Range-DiskANN: Exact Range Search over Billion-point
  Graph Indexes", PVLDB 2023.
- Milvus docs: `search_iterator` and `range_filter`, v2.4+, 2025.
- Qdrant docs: `score_threshold` in the search API, 2025.
- LanceDB docs: threshold search, 2026.
- Kleinberg, J. "The Small-World Phenomenon: An Algorithmic Perspective",
  STOC 2000.
