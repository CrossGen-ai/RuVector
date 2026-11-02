# IVF Centroid Locality Reorder

> Nightly research · 2026-11-02 · branch `research/nightly/2026-11-02-ivf-centroid-locality-reorder`

## Abstract

In an IVF index, a query probes the `nprobe` centroids closest to it, then
scans each of those clusters' posting lists. Posting lists are usually laid
out in **centroid-id order**, which is arbitrary (determined by the k-means
init). This work tests a cheap one-shot pre-processing step: *permute*
cluster ids so that ids that are close in centroid space are also close in
integer id space. The hope: when the query sorts its chosen probe ids before
scanning, consecutive probes hit adjacent memory regions, improving L1/L2
cache locality.

We implement two reorderings — a greedy nearest-neighbor chain
(`O(n_c^2 · d)`) and a 1-D random-projection sort (`O(n_c · d + n_c log n_c)`) —
measure probe-stride reduction and wall-clock speedup, and report the
honest result: on dim=128, 512 clusters, nprobe=24, speedup is ~1.01× with
0.6–0.9% stride reduction. The idea is cheap and correct but the regime
where it pays off is narrower than hoped. See *Practical failure modes*.

## SOTA survey

Reorder-for-locality is a mature idea in graph index literature but
underexplored for IVF:

- **FINGER (SIGMOD '23)** reorders HNSW neighbor lists for ADC-friendly
  layouts.
- **Reverse Cuthill–McKee** has been applied to HNSW adjacency arrays
  (e.g. `ruvector-hnsw-freq-reorder`) and sparse-matrix reorderings for
  decades.
- **Elias–Fano / StreamVByte adjacency** (`gist-elias-fano-hnsw-adjacency-2026`,
  `gist-streamvbyte-hnsw-adjacency-2026`) reduce *size* of neighbor lists;
  reorder reduces *stride* of consecutive accesses.
- **PDX / vertical layouts** (`gist-pdx-vertical-2026`) address the inner-loop
  layout; this work addresses the outer-loop (centroid walk) layout.
- **Faiss**, **Milvus**, **Weaviate**, **Qdrant**, **LanceDB**, **Pinecone**,
  **FAISS-IVF** all leave centroid ids in insertion/k-means-init order.
  To our knowledge no production IVF applies a locality permutation to the
  cluster-id axis.
- **ScaNN** uses hierarchical quantization but doesn't permute leaf ids
  for scan locality.

## Proposed design

1. Build an IVF index normally (random init k-means, 8 iterations).
2. Compute a permutation `new_from_old: [u32; n_clusters]` via a `Reorder`
   strategy (greedy chain or projection sort).
3. Rewrite the physical layout: `apply_permutation` produces a new `FlatIvf`
   where cluster `new_id` holds the vectors that used to live at
   `old_id = inv[new_id]`. The centroid array, offset array, flat vector
   buffer, and id vector are all permuted in lockstep.
4. At query time: rank centroids by distance (unchanged), take top-`nprobe`,
   then **sort the probe-id list ascending** before scanning. Sorted ids on a
   locality-reordered index means sequential probes read ascending offsets.

Design is a one-shot O(n_c² · d) preprocess for greedy chain, O(n_c · d) for
projection; both run once at build time and never again. No change to the
query API, index format, or recall.

## Implementation notes

Rust-only; crate `ruvector-ivf-centroid-locality-reorder`. The `Reorder`
trait is swappable, so additional strategies (Hilbert curve, OPTICS chain,
Christofides-TSP) drop in without touching `FlatIvf`. `Metric` is also a
trait — the current PoC uses squared L2 but cosine or inner-product kernels
plug in by implementing it.

Correctness is enforced by a property test (`reorder_matches_baseline_results`):
for a batch of queries, the top-5 sets from baseline and reordered indices
must be identical. The reorder is a pure permutation; it can never change
which vectors are returned.

## Benchmark methodology

- Platform: Darwin 24.6.0 (Apple Silicon), release build, LTO=thin,
  codegen-units=1.
- Dataset: 50 000 synthetic uniform points in `R^128`.
- Index: 512 IVF clusters, 8 k-means iterations.
- Queries: 1 000 synthetic uniform queries, nprobe=24, k=10.
- Metric: µs per query, averaged. Warm-up: 16 queries excluded from the timer.
- Probe stride: for each query, sum of `|offset[c_{i+1}] - offset[c_i]|` across
  the sorted probe list — a direct proxy for how clumped the memory accesses are.

Build cost is reported separately: greedy chain = 4.2 ms, projection sort = 0.1 ms,
once at build time.

## Results

All numbers from `cargo run --release --bin bench`:

```
Variant             |  µs / query | avg probe stride
--------------------+-------------+------------------
baseline            |     111.98  |           46016
greedy-chain        |     110.57  |           45744
projection-sort     |     111.42  |           45622

Speedup greedy vs baseline:    1.01x
Speedup projection vs baseline:1.01x
Stride reduction greedy:       0.6%
Stride reduction projection:   0.9%
```

Memory footprint is identical across variants (24.86 MiB) — reorder is a
permutation.

Verdict: **mechanism works, magnitude is small in this regime.** Reads are
real; the chain and projection both reduce stride; recall is unchanged
(property test enforces it). But the dominant cost of each probe is the
sequential L2 distance over ~100 128-D vectors inside a cluster, not the
jump between clusters, so cache locality on the inter-cluster seam is a
minor fraction of total time.

## How it works (walkthrough)

Think of an IVF index as a two-level walk: query → centroid shortlist →
posting lists. The *inner* walk (scanning one posting list) is already
sequential, so a lot of vector-DB performance work has focused there
(SIMD, FastScan, PDX vertical, cascade ADC). The *outer* walk — jumping
from one cluster's posting list to the next — is sequential-in-id but
*random-in-space*: cluster 7 and cluster 8 may be far apart in `R^d`, so
k-means produces post-build neighbors that end up stored far apart.

The trick: pre-sort the clusters so cluster 7 and cluster 8 are physical
neighbors. Then if a query probes clusters that happen to lie in the same
region of space (which is the common case — nearby clusters to a query are
nearby to each other), the sorted probe-id list traverses a tight arc of
the memory arena instead of scattering across it.

Why doesn't this collapse into a Hilbert curve? High-D data doesn't fit on
a Hilbert curve well (dimensionality curse), so a greedy chain or a 1-D
random projection are the practical options. The chain is tighter locally
but can "double back" on itself; the projection is cheaper and smoother on
average.

## Practical failure modes

- **Cluster size dominates probe cost.** At dim=128 × ~100 vectors each,
  one posting list is already ~50 KiB — larger than L1 and comparable to L2.
  Prefetching one cluster evicts the previous. Reorder helps only on the
  seam between posting lists, which is a small fraction of total scan time.
- **High `nprobe` dilutes locality.** The chain only guarantees adjacency
  between a cluster and its single nearest neighbor; with `nprobe=24`, most
  of the probed clusters aren't chain-neighbors anyway.
- **k-means init matters.** With a smarter init (k-means++ or hierarchical
  k-means), the natural centroid order already has some locality; the
  reorder has less to recover.
- **High dimension defeats the 1-D projection.** The projection-sort
  captures only one axis of variance; above ~32 dims it leaves most of the
  structure on the table. The chain is more robust but quadratic.
- **Streaming/updates invalidate the permutation.** Adding a new centroid
  reopens the question of where to insert it in the id axis. This PoC is
  one-shot build-time only.

## What to improve next (roadmap)

1. **Switch to narrow clusters (<=1 KiB each)**, e.g. 2 048 clusters × 25
   vectors × 128 dim = 12.8 KiB. Then one cluster fits in L1 and the
   inter-cluster seam becomes a larger fraction of total time. Expect
   speedup to climb to 1.05–1.10×.
2. **Replace probe-stride with cycle-counter perf.** The current stride
   proxy conflates physical distance with cache behavior. Linux perf
   counters for L1-dcache-load-misses would pin down whether the mechanism
   is really a cache win or just a prefetch-friendliness nudge.
3. **Combine with PQ ADC inner-loop**. If each cluster is a 16-byte PQ
   code per vector, 100 codes × 16 B = 1.6 KiB — comfortably in L1. In
   that regime, reorder should help substantially more, because the
   inter-cluster seam becomes the dominant stall.
4. **Hilbert-ish embeddings via multiple projections.** Project onto `k`
   random axes, build a Z-order, sort. Captures more variance than a
   single axis at `O(k · n_c · d)`.
5. **TSP solver on centroids.** Christofides or 2-opt on the all-pairs
   centroid distance matrix for a tighter chain than greedy. Still
   `O(n_c² · d)` for the matrix.
6. **Benchmark on real datasets** — SIFT1M, GIST, Deep1B — not synthetic
   uniform; the chain's benefit will vary with how clustered the real
   centroids are.

## Production crate layout proposal

```
crates/
├── ruvector-ivf-centroid-locality-reorder/
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                   # FlatIvf, Reorder trait, apply_permutation
│   │   ├── strategies/
│   │   │   ├── greedy_chain.rs      # O(n²·d), tight
│   │   │   ├── projection_sort.rs   # O(n·d + n log n), smooth
│   │   │   ├── hilbert_zorder.rs    # TODO: multi-axis z-order
│   │   │   └── tsp_two_opt.rs       # TODO: 2-opt refinement
│   │   └── bin/
│   │       └── bench.rs             # synthetic bench
│   ├── tests/
│   │   └── real_datasets.rs         # SIFT1M, GIST (behind feature flag)
│   └── benches/
│       └── criterion.rs             # criterion harness
```

## References

1. Jégou, Douze, Schmid — *Product Quantization for Nearest Neighbor Search*
   (TPAMI 2011).
2. Johnson, Douze, Jégou — *Billion-scale similarity search with GPUs*
   (Faiss, 2017).
3. Zhang et al. — *FINGER: Fast Inference for Graph-based Approximate
   Nearest Neighbor Search* (SIGMOD 2023).
4. Cuthill, McKee — *Reducing the bandwidth of sparse symmetric matrices*
   (ACM 1969).
5. Guo et al. — *Accelerating Large-Scale Inference with Anisotropic
   Vector Quantization* (ScaNN, ICML 2020).
6. Faiss source: `faiss/IndexIVF.cpp`, list storage.

## How to reproduce

```bash
cd crates/ruvector-ivf-centroid-locality-reorder
cargo test --release
cargo run --release --bin bench
```

Expected runtime: ~90 s build + ~10 s bench on a modern laptop.
