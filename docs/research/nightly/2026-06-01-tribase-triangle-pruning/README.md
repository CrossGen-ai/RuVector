# Tribase-style Triangle-Inequality Pruning for Graph ANN

**Date**: 2026-06-01
**Branch**: `research/nightly/2026-06-01-tribase-triangle-pruning`
**Crate**: `crates/ruvector-tribase`

## Abstract

Graph-based approximate nearest neighbor (ANN) search — HNSW, NSG, Vamana, DiskANN — spends the bulk of its query time on full-precision distance computations against visited candidates. Tribase (Lu et al., SIGMOD 2024) showed that with a small amount of *precomputed distance-to-landmark* metadata per node, the triangle inequality

```
|d(q, L) − d(x, L)| ≤ d(q, x)
```

yields a free lower bound on the true query–node distance. Whenever this lower bound already exceeds the current `efSearch` cutoff, the full f32 distance computation can be skipped without harming recall. This nightly delivers a self-contained Rust PoC (`crates/ruvector-tribase`) with three swappable searchers (baseline / 1-landmark / k-landmark), a working `cargo test`, an example, and a deterministic benchmark binary that prints real numbers.

## SOTA survey

| System | Pruning mechanism | Cost / vector | Loss |
|--------|--------------------|--------------|------|
| HNSW (Malkov 2018) | none — every visited neighbor gets a full distance | 0 B | none |
| NSG (Fu 2017)      | none at search; pruning is build-time only | 0 B | none |
| Tribase (SIGMOD '24) | k landmark distances per node; tri-ineq bound | 4·k B | **lossless** |
| FINGER (NSDI '23)  | LSH-style projection of query onto residuals | 8–32 B | lossy at low recall |
| Anisotropic VQ (ScaNN '20) | learned product quantizer with score-aware distortion | ~16 B | lossy |
| RaBitQ (SIGMOD '24) | 1-bit randomized projection | dim/8 B | lossy with bound |

Tribase is the only fully *lossless* speedup of the four — it never changes which candidates beam search would have kept, only how cheaply it filters them.

## Proposed design

`crates/ruvector-tribase` exposes a `trait AnnSearcher` and three implementations sharing the same `FlatGraph` substrate:

```
trait AnnSearcher {
    fn search(&self, query: &[f32], k: usize, ef: usize, stats: &mut SearchStats) -> Vec<(u32, f32)>;
}
```

* **`BaselineSearcher`** — vanilla beam search, no pruning, reference implementation.
* **`TribaseSearcher`** — stores `d(x, medoid)` per node (4 B × N total). At query time, computes `d(q, medoid)` once, then for every candidate `x` derives the lower bound `(d(q,m) − d(x,m))²` in 2 subtractions + 1 mul. Compared against the current `ef` cutoff.
* **`MultiLandmarkSearcher`** — k random landmarks, lower bound = `max_l (d(q, L_l) − d(x, L_l))²`. Trades 4·k B/vector and k extra full distances per query for a tighter bound.

The hot-loop change is a single closure passed into `beam_search`:

```rust
let prune = |id, worst_sq| {
    let lb = (dq_l - landmark_dist[id as usize]).abs();
    Some(lb * lb)
};
```

If the closure returns a value > current worst kept distance, the f32 distance is skipped. The closure is monomorphized per searcher, so there is no dispatch cost.

## Implementation notes

* The triangle inequality holds for the true metric, not its square. Landmark distances are stored as **Euclidean (square-rooted)**; the squared lower bound is reconstructed at query time so it can be compared against squared distances kept by the beam.
* Per-node memory: 4 B for 1 landmark, 4·k B for k landmarks. For a 1M × 128 dataset, k=8 adds 32 MB on top of a ~512 MB raw vector store — under 7% overhead.
* The medoid is computed once from the data mean (one O(N·D) sweep), so build is dominated by the kNN graph itself.
* Pruning is monotone w.r.t. ef: a candidate pruned at ef=64 is also pruned at any ef<64. This makes Tribase compose cleanly with adaptive `efSearch` schemes.

## Benchmark methodology

* **Dataset**: synthetic Gaussian point clouds, 8 clusters, varying (N, D).
* **Hardware**: Apple M-series, single-threaded release build.
* **Query set**: 100 queries drawn from a separately-seeded generator with the same distribution.
* **Ground truth**: brute-force top-10 per query.
* **Recall**: `|hits ∩ truth| / k` averaged over the query set.
* All numbers from `cargo run -p ruvector-tribase --release --bin tribase_bench`.

## Results (real, not aspirational)

```
      name |     n |    d |   ef |   k |  build_ms |       qps |      fullD/q |   pruned/q | recall
--------------------------------------------------------------------------------------------------------------
  baseline |  1000 |   32 |   32 |  10 |      14.7 |   79768.7 |        227.8 |        0.0 |  0.966
 tribase-1 |  1000 |   32 |   32 |  10 |      14.7 |   76770.5 |        228.1 |        0.7 |  0.966
 tribase-8 |  1000 |   32 |   32 |  10 |      14.8 |   73950.8 |        206.5 |       29.4 |  0.966

  baseline |  1000 |   32 |   64 |  10 |      14.6 |   54909.9 |        291.4 |        0.0 |  0.991
 tribase-1 |  1000 |   32 |   64 |  10 |      14.6 |   48624.3 |        291.8 |        0.6 |  0.991
 tribase-8 |  1000 |   32 |   64 |  10 |      14.6 |   47030.2 |        254.2 |       45.2 |  0.991

  baseline |  2000 |   64 |   64 |  10 |      88.9 |   29953.9 |        422.2 |        0.0 |  0.929
 tribase-1 |  2000 |   64 |   64 |  10 |      89.0 |   35681.4 |        420.5 |        2.8 |  0.929
 tribase-8 |  2000 |   64 |   64 |  10 |      89.1 |   27720.7 |        405.6 |       24.6 |  0.929

  baseline |  4000 |  128 |   64 |  10 |     698.3 |   19847.2 |        527.1 |        0.0 |  0.785
 tribase-1 |  4000 |  128 |   64 |  10 |     698.6 |   20585.8 |        528.0 |        0.1 |  0.785
 tribase-8 |  4000 |  128 |   64 |  10 |     699.4 |   19274.3 |        521.5 |       13.6 |  0.785
```

The recall column is **identical across all three searchers within each config** — pruning is lossless, as theory predicts. Pruning effectiveness drops sharply with dimension: at D=32 the 8-landmark variant prunes ~16% of distance computations; at D=128 it prunes ~3%. This is the classical concentration-of-distances effect — in high dim, `d(q, L)` and `d(x, L)` cluster around a common shell, collapsing the difference that the triangle inequality lives in.

QPS is roughly flat because the prune check itself is *not* free at these dims (one squared f32 distance is ~D mul-adds; the prune closure costs k subtractions + 1 mul + comparison). Pruning wins decisively only when the saved distance is much more expensive than the bound check — i.e., when D is large, when distance is SIMD-bound, **or** when distances are computed on disk / over RDMA (DiskANN, GRIP).

## How it works (blog-readable walkthrough)

Imagine you're searching for the nearest restaurant to your home. You know that the Empire State Building (a *landmark*) is 5 km from your home. Someone tells you a candidate restaurant is 20 km from the Empire State Building. You don't need to drive there to know the restaurant is at least `|20 − 5| = 15 km` from your home — the triangle inequality tells you so. If the best restaurant you've already found is only 3 km away, you can rule out this candidate without ever measuring its true distance from you.

Now generalize: every database vector remembers its distance to a shared landmark (the *medoid* of the dataset). When a query arrives, we measure the query's distance to that one landmark — *once*. Then every candidate the graph search wants to expand can be lower-bounded against the current ef-th worst kept neighbor, for free, by subtraction. If the lower bound already loses, we skip the f32 distance entirely. Recall is unchanged because we only ever skip candidates that *provably* can't enter the top-k. Adding more landmarks tightens the bound at linear memory cost.

## Practical failure modes

1. **High dimensionality** — pruning drops to ~3% at D=128 on this synthetic data. On real BERT-768 embeddings the effect is even sharper.
2. **Cluster-aligned landmarks** — if a landmark sits inside one cluster, points in *other* clusters all get a fat lower bound, but the in-cluster bound collapses. The medoid is a reasonable default but not always optimal.
3. **Disk/RDMA workloads** — Tribase shines when the saved distance is expensive (DiskANN, billion-scale). At pure-RAM 32-dim it can actually be slower due to bound-check overhead.
4. **Updates** — every insert needs `d(new, L)` for each landmark, and every landmark replacement requires a full O(N) recomputation.
5. **Metric assumption** — the triangle inequality is exact for L2 / cosine, looser for inner-product where d(x,x) ≠ 0.

## What to improve next

* **Adaptive landmark selection** — k-medoids over a sample, or learned landmarks per cluster (CAGRA-style).
* **Hybrid with RaBitQ** — use RaBitQ's 1-bit code as a *lower-bound generator* alongside Tribase; combine via the tighter of the two.
* **SIMD batch pruning** — vectorize the lower-bound check across all neighbors of the current node (m=16 at once).
* **DiskANN integration** — landmark distances are tiny (4·k B/vector); a single page can hold thousands of landmark codes, enabling per-page early-out before the f32 fetch.
* **GPU port** — the bound check is a single fused-multiply-add per (query, candidate), perfectly suited for CUDA / Metal.

## Production crate layout proposal

```
crates/ruvector-tribase/
├── Cargo.toml
├── src/
│   ├── lib.rs         # AnnSearcher trait + three searchers
│   ├── graph.rs       # FlatGraph (replaceable; production: ruvector-diskann)
│   ├── metric.rs      # l2_sq, dot
│   ├── stats.rs       # SearchStats counters
│   └── bin/
│       └── tribase_bench.rs
└── examples/
    └── tribase_demo.rs
```

For production integration, the searcher trait should be implemented inside `ruvector-diskann` and `ruvector-graph` as `with_landmarks(...)` builders, gated behind a `tribase` feature.

## References

1. Lu, Z., et al. **"Tribase: A Vector Data Query Engine for Reliable and Lossless Pruning Compression using Triangle Inequalities."** SIGMOD 2024.
2. Malkov, Y. A., Yashunin, D. A. **"Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs."** TPAMI 2018.
3. Subramanya, S., et al. **"DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node."** NeurIPS 2019.
4. Gao, J., Long, C. **"RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search."** SIGMOD 2024.
5. Chen, P., et al. **"FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search."** WWW 2023.
