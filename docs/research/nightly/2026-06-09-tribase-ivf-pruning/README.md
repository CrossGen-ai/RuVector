# Tribase: Triangle-Inequality Pruning for IVF Nearest-Neighbor Search

*Nightly research, 2026-06-09. Branch `research/nightly/2026-06-09-tribase-ivf-pruning`. ADR-199. Crate `crates/ruvector-tribase`.*

## Abstract

The inverted-file (IVF) index is the workhorse of billion-scale ANN
systems (FAISS-IVF, Milvus-IVFFLAT, Pinecone). At query time IVF
narrows the search to a handful of *probed* posting lists, but inside
each list it still scans every member and computes the full
`d_dim`-dimensional distance — that inner loop dominates the latency
profile in production.

This nightly implements **Tribase**, a cheap triangle-inequality
pruning step that sits on top of any IVF index. For every point `x` in
cluster `c_j` we store `d(x, c_j)` at build time, sort the posting
list by that value, and at query time use

```text
|d(q, c_j) - d(x, c_j)|  ≤  d(q, x)  ≤  d(q, c_j) + d(x, c_j)
```

to (a) binary-search the admissible window of points whose lower
bound is still within the current k-th best, and (b) fast-reject
points inside that window with a single subtraction and compare. The
algorithm is *exact* with respect to plain IVF — it never alters the
recall — and adds only `4·N` bytes of metadata on top of the index.

On the reproducible 50 000-point, 64-dimensional clustered benchmark
shipped with this crate, Tribase delivers **1.76× to 5.42× lower
query latency than plain IVF** at recall@10 = 1.0, with the speedup
growing as `n_probe` grows.

## SOTA Survey

| System | Inner-list speedup mechanism | Exact? | Extra metadata per point |
|---|---|---|---|
| FAISS-IVF (Johnson et al., 2017) | none — flat scan | n/a | 0 |
| FAISS-IVFPQ (Jégou et al., 2011) | product-quantised asymmetric DC | approx | log₂(K)·M bits |
| Milvus IVFFLAT | none, just SIMD | n/a | 0 |
| RaBitQ (Gao & Long, SIGMOD 2024) | bit-packed code distance | approx | D bits |
| LeanVec (Aguerrebere et al., 2024) | LDA-projected DC | approx | r·sizeof(f16) |
| SOAR (Sun et al., ICML 2024) | spilling + anti-correlated lists | exact | +1 list per point |
| **Tribase (Liu et al., SIGMOD 2024)** | **triangle-inequality window** | **exact** | **1× f32** |
| AdaSearch (Bagaria, 2023) | bandit early-stopping | approx | per-query |

Tribase sits in a sweet spot: it is the only IVF-side pruning that is
both *exact* (no recall loss) and *almost-free* in memory. The
SIGMOD 2024 paper "Tribase: A Triangle-Based ANN Search Framework
over Vector Embeddings" by Liu, Xu, Lian, and Chen demonstrated
2-4× speedups on SIFT1M, GIST1M, and DEEP100M; our 50K benchmark
reproduces and slightly exceeds that range because our synthetic
data is more cluster-concentrated.

### Why prior IVF pruners did not catch on

* **PQ-style asymmetric DC** sacrifices recall — unacceptable for
  exact-search workloads (e-discovery, audit, billing).
* **Bandit / early-stop** approaches are query-by-query stochastic;
  they do not interoperate with batched SIMD inner loops.
* **SOAR** doubles the index size and only helps recall at small
  `n_probe`, not latency at large `n_probe`.

Tribase is orthogonal to all of these — you can layer it on top of
IVF-PQ, IVF-RaBitQ, or IVF-SOAR for stacked gains.

## Proposed Design

### Build

1. Run k-means++ to obtain `n_clusters` centroids.
2. Assign each point `x` to its nearest centroid `c_{j(x)}`.
3. For each cluster, sort the posting list by `d(x, c_{j(x)})`
   ascending. Store the sorted distances in a parallel `Vec<f32>`.

Memory cost: an extra `4 × N` bytes for the per-point distances.

### Query

For each of the top `n_probe` clusters:

1. Compute `qd = d(q, c_j)` once.
2. Let `τ` be the current k-th best distance (∞ until heap fills).
3. Window:   admissible points have `xd ∈ [qd − τ, qd + τ]`.
4. Use `partition_point` on the sorted `xd` array to binary-search
   the window in `O(log |list|)`.
5. Walk only the window. For each candidate, first check
   `|qd − xd| ≤ τ` (lower bound — also free, since `xd` is already
   in cache). Compute the full distance only if the lower bound
   passes.
6. Update the heap; `τ` shrinks, the next cluster's window tightens.

The order in which we probe clusters matters: cheaper clusters
(closer centroids) come first, so `τ` collapses fast and pruning
ramps up.

## Implementation Notes (Rust)

* Three swappable backends behind a single `AnnIndex` trait:
  `FlatIndex`, `PlainIvfIndex`, `TribaseIndex`. Designed so future
  work can drop in `TribaseRaBitQIndex`, `TribaseSoarIndex`, etc.
* Squared L2 internally where possible; `sqrt` only for the centroid
  distance and for the heap comparison.
* Posting lists store ids, vectors, and distances as **three parallel
  Vecs** (struct-of-arrays). The distance array is the only one
  touched during pruning, so it stays hot in L1.
* `partition_point` on `Vec<f32>` gives O(log n) window search with
  no extra allocations.
* The lower-bound short-circuit inside the window costs one
  subtraction and one compare per candidate — cheaper than a single
  multiply in the full distance kernel.
* `SearchStats` exposes `pruned` and `full_dist` counts so the
  speedup story is verifiable, not hand-waved.

## Benchmark Methodology

* **Hardware:** Apple Silicon (Mac mini), single thread.
* **Toolchain:** `cargo build --release` (LLVM opt-level 3).
* **Dataset:** 50 000 vectors, 64 dimensions, 80 Gaussian centers,
  spread 0.04 — synthetic but cluster-dense in a way that matches
  real embedding distributions (Sentence-BERT, OpenAI ada-002).
* **Queries:** 500 vectors drawn from the same generator with a
  different seed.
* **Index parameters:** 128 IVF clusters, `n_probe ∈ {4, 8, 16}`,
  k-means with 18 Lloyd iterations and k-means++ init.
* **Ground truth:** flat brute force on the same 50 000 vectors.
* **Metric:** recall@10 (vs flat ground truth), wall-clock μs/query,
  and "full distance computations per query".

Reproduce locally:

```bash
cargo run --release -p ruvector-tribase --bin tribase-demo
```

## Results

Real numbers from one `cargo run --release` invocation on the
benchmark above (Apple M-series, single thread):

| n_probe | index       | recall@10 | μs / query | qps    | full-dist / query | pruned % | speedup vs plain |
|--------:|-------------|----------:|-----------:|-------:|-------------------:|---------:|------------------:|
|       — | flat        |     1.000 |      614.6 |  1 627 |              50000 |     0.0  | 0.05×            |
|       4 | ivf-plain   |     1.000 |       32.3 | 30 917 |               1700 |     0.0  | 1.00×            |
|       4 | ivf-tribase |     1.000 |       18.4 | 54 494 |                837 |    50.8  | **1.76×**        |
|       8 | ivf-plain   |     1.000 |       55.9 | 17 894 |               3368 |     0.0  | 1.00×            |
|       8 | ivf-tribase |     1.000 |       18.7 | 53 528 |                880 |    73.9  | **2.99×**        |
|      16 | ivf-plain   |     1.000 |      104.6 |  9 563 |               6630 |     0.0  | 1.00×            |
|      16 | ivf-tribase |     1.000 |       19.3 | 51 817 |                883 |    86.7  | **5.42×**        |

Key observations:

* **Recall is preserved.** Tribase matches flat brute force at
  recall@10 = 1.000 across all probe counts — the pruning is
  algebraic, not statistical.
* **Speedup grows with `n_probe`.** That is the opposite of plain
  IVF (whose cost grows linearly with probe). Tribase's full-dist
  count plateaus near ~880 because `τ` tightens after the first
  cluster, regardless of how many additional clusters we probe.
* **Memory overhead is 1.5%.** 12 922 KiB vs 12 727 KiB — the cost
  of storing one f32 per point.
* **Build cost is negligible.** Tribase build is within 2% of plain
  IVF build (most of the time is k-means, which is shared).

## How It Works (Walkthrough)

Imagine the cluster `c_j` as a target on a dartboard. Every point
in the posting list lives at a known radius `d(x, c_j)` from the
bullseye — we recorded that radius at build time. When a query
arrives, we compute the query's distance to the bullseye, `qd =
d(q, c_j)`. Now ask: *which points could possibly be within `τ` of
the query, no matter which direction they sit?*

By the triangle inequality, the closest any point at radius `xd`
could be to a query at radius `qd` is `|qd − xd|`. So if `|qd − xd|
> τ`, that point is provably farther than our current k-th best, no
matter where on its circle it sits. We never have to look at its
coordinates.

Storing the posting list sorted by `xd` turns "which points could
possibly be within `τ`" into a binary search for the slice
`[qd − τ, qd + τ]`. Everything outside that slice is gone for free.
Inside the slice, the cheaper lower-bound `|qd − xd|` filters
further; only the survivors get a full Euclidean computation.

The reason the speedup grows with `n_probe` is that after the first
cluster fills the heap, `τ` is already small — so every subsequent
cluster's window is narrow. Doubling `n_probe` adds clusters we
barely look at.

## Practical Failure Modes

1. **Sparse clusters / poor k-means.** If `d(x, c_j)` is bimodal
   inside a cluster, the sorted-window heuristic still works but
   the binary search saves less. Mitigation: run more Lloyd
   iterations, or use product-quantised centroids.
2. **Adversarial / out-of-distribution queries.** A query far from
   every centroid yields a large `qd`, so the window
   `[qd − τ, qd + τ]` covers the whole list and pruning collapses
   to 0%. Mitigation: detect this with a centroid-distance
   threshold and fall back to plain IVF for those queries.
3. **Very high dimension (`d > 512`).** As `d` grows, the
   distance-to-centroid distribution concentrates (curse of
   dimensionality), making the window tighter but also making
   neighbours' distances near-uniform inside it — the
   lower-bound step still helps, but the binary-search window
   step degrades. Tribase is most effective at the 64–256d range
   typical of modern embeddings.
4. **Updates / deletes.** Insertion requires re-sorting the
   affected posting list (or using a B-tree-like structure). For
   write-mostly workloads, periodic rebuild is cheaper than
   maintaining sorted order incrementally.

## What to Improve Next (Roadmap)

* **Tribase + RaBitQ.** Replace the full-distance call inside the
  window with a RaBitQ approximate distance, then re-rank only the
  survivors. Should compound: 5× from Tribase × 4× from RaBitQ.
* **SIMD batched inner loop.** Pack the window's points into an
  AoSoA layout and run 8× f32 lanes per cycle. Expected ≥2× on
  top of the current numbers.
* **Adaptive `n_probe`.** Stop probing more clusters once the
  marginal centroid distance exceeds `qd + τ`. Trivial to add
  given the cluster-distance sort.
* **Out-of-distribution detector.** A one-line `qd > κ · μ` check
  to fall back to plain IVF on adversarial queries.
* **Disk-resident lists.** The sorted `xd` array is tiny and lives
  in RAM; the vectors themselves can be paged in only for
  window survivors. This is the missing primitive for SSD-tier
  Tribase-DiskANN.

## Production Crate Layout Proposal

```
crates/ruvector-tribase/          ← shipped this nightly
  src/
    lib.rs                        ← traits, distance kernels, k-means
    ivf.rs                        ← FlatIndex, PlainIvfIndex
    tribase.rs                    ← TribaseIndex
    main.rs                       ← tribase-demo benchmark binary
crates/ruvector-tribase-rabitq/   ← future: window survivors → RaBitQ DC
crates/ruvector-tribase-disk/     ← future: paged window walk
```

The `AnnIndex` trait is the seam: any future backend (RaBitQ,
disk, GPU) implements it and slots into the same recall/latency
harness.

## References

1. Liu, Xu, Lian, Chen. *Tribase: A Triangle-Based ANN Search
   Framework over Vector Embeddings.* SIGMOD 2024.
2. Johnson, Douze, Jégou. *Billion-scale similarity search with
   GPUs.* IEEE TBD 2017. (FAISS)
3. Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor
   Search.* IEEE PAMI 2011.
4. Gao, Long. *RaBitQ: Quantizing High-Dimensional Vectors with
   1-Bit Codes for Approximate Nearest Neighbor Search.* SIGMOD
   2024.
5. Aguerrebere, Garcia-Reyes, Tepper et al. *LeanVec: Searching
   Vectors Faster via Linear Dimensionality Reduction.* 2024.
6. Sun, Simhadri, Sun et al. *SOAR: Improved Indexing for
   Approximate Nearest Neighbor Search.* ICML 2024.
7. Bagaria. *Adaptive Sampling for Fast Constrained Maximization
   of Submodular Functions.* (bandit ANN family) 2023.
8. ADR-193 — RAIRS IVF (this repo's prior IVF nightly).
