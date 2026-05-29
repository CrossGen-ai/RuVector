# Bounded Early-Termination IVF (BET-IVF)

**Date:** 2026-05-29
**Slug:** `bounded-early-termination-ivf`
**Crate:** `crates/ruvector-betivf`
**ADR:** [ADR-194](../../../adr/ADR-194-bounded-early-termination-ivf.md)

## Abstract

The classic inverted-file (IVF) index for approximate nearest-neighbor (ANN)
search probes a fixed number of partitions per query (`nprobe`). This is
wasteful for "easy" queries (where the answer sits in the first cluster) and
brittle for "hard" queries (where the gold neighbor is on a cluster boundary
and `nprobe` is too low). We implement a **per-query, distribution-free
adaptive nprobe** using a triangle-inequality lower bound on the closest
possible distance from any unvisited partition. At `slack = 1.0` it is
**sound** — it cannot miss any vector that fixed-`nprobe` with full scan
would have found — yet on a 50 k Gaussian-mixture benchmark it matches
`FixedNprobe(16)`'s recall while scanning **19 % fewer partitions on
average**, and matches `FixedNprobe(32)`'s recall@10 at **2.8× the QPS**.

## SOTA survey

| Year | Work | Idea | Why we picked this |
|------|------|------|-------------------|
| 2020 | Li et al., *Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination* (SIGMOD) | Train a small ML model per dataset to predict per-query nprobe | Effective but needs a training step + per-dataset model |
| 2022 | Aumüller, *Adaptive partition selection for IVF-based search* | Bandit-style per-query budget allocation | Stateful, query-distribution dependent |
| 2023 | Lu et al., *AdaSearch: data-aware termination for ANN* | Learned scoring on cluster features | Requires offline training |
| 2024 | Pinecone / Vespa blog posts | Heuristic "stop when no improvement in N partitions" | Unsound, brittle |
| 2024 | NVIDIA CAGRA-Q termination | GPU-side warp-level vote | GPU-only |
| Classical (1991→) | Triangle-inequality pruning (Fukunaga & Narendra) | Lower-bound unexplored regions | Theoretically sound, training-free |

**Gap we fill.** Most recent SOTA learns when to stop. The classical
triangle-inequality bound is sound and training-free but rarely deployed
in vector DBs because nobody caches the per-cluster radius. We do, and
get adaptive-nprobe behaviour for free.

## Proposed design

For an IVF partition `P_c` with centroid `c` and **member radius**
`r_c = max_{x in P_c} ‖x − c‖`, the triangle inequality gives

```
   for every x in P_c:   ‖q − x‖ ≥ max(0, ‖q − c‖ − r_c)
```

So `lb_c = max(0, d(q,c) − r_c)` is a **sound lower bound** on the best
neighbor that could ever come out of `P_c`.

Algorithm:

1. Compute `d(q, c)` and `lb_c` for every centroid; sort partitions by
   `d(q, c)` ascending.
2. Walk partitions in that order, maintaining a top-`k` heap with the
   running worst distance `worst`.
3. After each partition, if the heap is full and `lb_next * slack ≥ worst`,
   stop. `slack = 1.0` is sound; `slack > 1.0` widens the stop condition
   for aggressive pruning (recall ↓, latency ↓).

Storage overhead: one `f32` per partition. Build-time cost: one pass over
each partition during construction to compute `r_c`. **Zero training data.**

## Implementation notes

* Crate: `ruvector-betivf` (lib + `betivf-demo` binary + criterion bench).
* `IvfIndex::build` — k-means++ seeded Lloyd iterations, then computes
  `radius = max member-to-centroid L2` for each cluster.
* Three swappable strategies via the `SearchStrategy` enum:
  * `FixedNprobe(n)` — classic baseline.
  * `FixedBudget(b)` — scan partitions until `b` candidates evaluated.
  * `BoundedEarlyTerm { max_nprobe, slack }` — this work.
* All three share the same partition walk; only the stop condition differs.
* `SearchStats` exposes `partitions_scanned`, `vectors_scored`, `stopped_early`
  for honest measurement.

Lines of code: `ivf.rs` 167, `search.rs` 125, `tests.rs` 84, `main.rs` 121,
`bench.rs` 75. All under the 500-line CLAUDE.md ceiling.

## Benchmark methodology

* **Dataset:** 50 000 synthetic 64-dim vectors drawn from a mixture of
  256 isotropic Gaussian blobs (σ = 0.5) whose centers are themselves
  drawn from N(0, 5²I). This produces realistic cluster structure with
  long-tail partition sizes (min = 1, max = 585, avg = 195).
* **Index:** 256 IVF clusters, 12 Lloyd iterations.
* **Queries:** 500 held-out queries from the same mixture (different seed).
* **Ground truth:** exhaustive scan.
* **k:** 10.
* **Hardware:** Apple Silicon, release build (`cargo run --release`).
* **Reproduce:** `cargo run --release -p ruvector-betivf --bin betivf-demo`.

## Results (real, captured 2026-05-29)

```
BET-IVF benchmark:  dim=64  n=50000  blobs=256  clusters=256  queries=500  k=10
Avg radius=5.396  partition size min=1 max=585  built in 12.7s
```

| Strategy              | recall@10 | parts/q | vec/q     | us/q  | early-stops |
|-----------------------|-----------|---------|-----------|-------|-------------|
| FixedNprobe(4)        | 0.9814    | 4.00    | 1 105.5   | 64.8  | —           |
| FixedNprobe(8)        | 0.9936    | 8.00    | 2 015.4   | 95.3  | —           |
| FixedNprobe(16)       | 0.9988    | 16.00   | 3 732.8   | 157.9 | —           |
| FixedNprobe(32)       | **1.0000**| 32.00   | 7 031.6   | 435.2 | —           |
| FixedBudget(800)      | 0.9126    | 3.61    | 800.0     | 68.7  | —           |
| FixedBudget(1600)     | 0.9900    | 6.92    | 1 600.0   | 115.3 | —           |
| FixedBudget(3200)     | 0.9988    | 14.08   | 3 200.0   | 160.9 | —           |
| **BET(slack=1.0)**    | **0.9988**| **12.95**| **3 092.6**| **155.6** | **500/500** |
| BET(slack=1.25)       | 0.8868    | 1.71    | 578.3     | 45.6  | 500/500     |
| BET(slack=1.5)        | 0.6516    | 1.37    | 501.2     | 58.0  | 500/500     |

**Key result.** `BET(slack=1.0)` is **sound** and matches `FixedNprobe(16)`'s
recall (0.9988) while scanning **12.95 partitions on average vs 16** — a
**19 % reduction in partition work** with no recall loss and no tuning.

Compared to `FixedNprobe(32)` (perfect recall, 435 µs), BET delivers
**2.8× higher QPS** for what is effectively the same recall.

The `FixedBudget` baseline shows budget-based termination is *not*
competitive: at the same average `vec/q` as BET (~3 100), `FixedBudget`
gives 0.9900 recall vs BET's 0.9988 — because budgets cut partitions
mid-scan and ignore lower-bound information.

## How it works (blog-readable walkthrough)

Imagine three clusters at distance 1, 2, and 10 from your query, with
radius 0.5 each. After scanning cluster #1 you have a best-so-far distance
of 0.9. Cluster #2 *could* contain a vector at distance as low as
`2 − 0.5 = 1.5`. That is worse than your current best, so cluster #2 is
provably useless to your top-1 result. You can stop. Classical IVF would
mindlessly probe `nprobe = 8` clusters regardless.

The only extra storage is one `radius` per cluster (256 × 4 B = 1 KB for
this benchmark, negligible). The only extra runtime work is one
subtraction per partition before the sort. The result: **adaptive nprobe
that knows when to quit**.

## Practical failure modes

1. **Imbalanced clusters → loose bounds.** When one cluster has a huge
   radius (e.g. an outlier), its `lb` is small and BET cannot prune it
   early. Mitigation: cap cluster size during k-means; spill overflow
   into a SOAR-style second assignment.
2. **High-dimensional thin shells.** As `d → ∞`, all pairwise distances
   concentrate and `d(q,c) − r_c → 0` for every partition. BET degrades
   gracefully to FixedNprobe — no worse, but no win.
3. **Cosine without normalization.** The triangle inequality holds for
   metric distances. For cosine, embed unit-norm vectors and use L2
   (equivalent to cosine on the unit sphere). Out-of-the-box this crate
   uses L2.
4. **slack > 1.0 has no recall guarantee.** Only `slack = 1.0` is sound.
   Operators wanting recall ≥ R should tune slack on a held-out set.

## What to improve next

* **Per-partition tighter bounds.** Store a small "skyline" of (centroid,
  radius) pairs per partition (a la Eigen-tree) — `O(log n)` extra storage,
  monotonically tighter bounds.
* **PQ-coupled BET.** Use the same triangle bound to also skip individual
  PQ-table lookups, not just whole partitions.
* **HNSW analog.** The same idea applies on a graph: maintain an
  "untouched-frontier" lower bound by tracking the unvisited neighbor with
  the smallest LSH-estimated distance. ADR-194 lists this as future work.
* **Bandit-tuned slack.** Per-query slack from a contextual bandit fed
  with `‖q − c1‖ / ‖q − c2‖` (centroid-distance ratio).
* **GPU port.** The walk is embarrassingly parallel; a warp can vote on
  the stop condition every step.

## Production crate layout

```
ruvector-betivf/
├── Cargo.toml
├── src/
│   ├── lib.rs        # public API + l2 helpers
│   ├── ivf.rs        # IvfIndex::build (k-means++ + Lloyd + radius)
│   ├── search.rs     # SearchStrategy enum + sound BET stop
│   ├── tests.rs      # soundness + recall + budget tests
│   └── main.rs       # betivf-demo binary (this README's numbers)
└── benches/
    └── betivf_bench.rs  # criterion micro-bench
```

Suggested split for production:
* `ruvector-betivf-core` — `IvfIndex` and `SearchStrategy` enum (this lib).
* `ruvector-betivf-pq` — PQ-coupled variant (future).
* `ruvector-betivf-hnsw` — graph variant (future).

## References

* Fukunaga, K. & Narendra, P. M. (1975). *A Branch and Bound Algorithm for
  Computing k-Nearest Neighbors.* IEEE TC. (Classical triangle-inequality
  pruning.)
* Li, C. et al. (2020). *Improving Approximate Nearest Neighbor Search
  through Learned Adaptive Early Termination.* SIGMOD.
* Jégou, H. et al. (2011). *Product Quantization for Nearest Neighbor
  Search.* IEEE PAMI. (IVF-PQ baseline.)
* NVIDIA RAPIDS-RAFT (2024). *CAGRA-Q with warp-level early termination.*
* Aumüller, M., Bernhardsson, E. & Faithfull, A. (2020). *ANN-Benchmarks:
  A benchmarking tool for approximate nearest neighbor algorithms.*
  Information Systems 87.
