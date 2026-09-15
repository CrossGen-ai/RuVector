# PDX: Dimension-Partitioned Vector Layout for Prunable ANN Scans

**Nightly research — 2026-09-15**
Slug: `pdx-vertical-vector-layout`
Crate: [`crates/ruvector-pdx-scan`](../../../../crates/ruvector-pdx-scan)
ADR: [ADR-346](../../../adr/ADR-346-pdx-vertical-vector-layout.md)

## Abstract

Ruvector's brute-force and IVF-list distance scans currently use a
row-major "horizontal" layout: each vector's `d` float32 dimensions live
contiguously in memory, and the L2 kernel loops over `d` inside a per-
vector reduction. The PDX paper (SIGMOD 2024, Kuffo et al.) proposes a
"vertical" alternative: pack vectors into blocks of `B` (typically 64) and,
inside a block, store dimension `j` for **all `B` vectors contiguously** —
a `d × B` column-major tile per block. Distance kernels then walk column-
by-column, keeping `B` per-lane accumulators and, optionally, comparing
partial distances against the running top-k threshold to prune doomed
lanes early.

This crate is a from-scratch Rust implementation with four scan variants
(horizontal baseline, PDX plain, PDX per-lane-pruned, PDX branchless
block-pruned), a k-means IVF driver stacked on either layout, and a
real-numbers benchmark. Measured on 50 000 × 128 Gaussian vectors, PDX
plain gives a **2.94× brute-force speedup at recall = 1.000**. Per-lane
pruning is a **net loss** (0.34×) because the per-lane predicate kills
Rust's autovectorization; a branchless block-level early-exit variant
recovers 1.86×. On IVF workloads with small nprobe, the threshold
tightens too slowly for pruning to pay for its overhead. These honest
tradeoffs — and where PDX wins vs. where it does not — are the point.

## SOTA survey

**PDX.** Kuffo, L.; Krommydas, E.; Boncz, P. "PDX: A Data Layout for
Vector Similarity Search." SIGMOD 2024. Introduces the dimension-major
block layout and shows 2–4× brute-force speedups on IVF partitions with
per-lane pruning under SIMD. The paper's own gains come primarily from
AVX-512 masked stores that make lane-level pruning branchless — a facility
absent from portable Rust without `std::simd` or `packed_simd`.

**Vector column stores.** DuckDB-vss and Milvus's segment format both keep
vectors row-major inside a "list" or "segment"; they rely on SIMD kernels
that reduce across `d`. Qdrant since 2024 experiments with "block-quantized"
storage but keeps horizontal layout for float32.

**Prunable distance kernels.** ADSampling (Gao & Long, SIGMOD 2023) proves
that a random projection subset of dimensions yields a distance lower bound
with tunable error, enabling pruning proportional to remaining dimensions;
FINGER (SIGMOD 2023) applies angle-based bounds specifically to HNSW
neighbor lists. PDX is complementary: same pruning intuition, applied at
the layout level so the pruned dimensions are literally never loaded.

**FAISS/cuVS.** Both keep horizontal storage for exact and PQ codes and
lean on massively-parallel GPU reductions. PDX is a CPU-first idea; the
paper reports no GPU numbers, and there is no cuVS PDX kernel today.

**Rust ecosystem.** `hnsw_rs`, `instant-distance`, `qdrant`, `usearch-rs`,
`lance` — all row-major. No PDX implementation exists in Rust as of the
2026-09 crates.io snapshot. This crate appears to be the first.

## Proposed design

Two backends implementing a common `VectorStore` trait:

```rust
pub trait VectorStore {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn get(&self, id: usize, out: &mut [f32]);
}
```

* `HorizontalStore` — flat `n * d` `Vec<f32>`. Baseline; also the input to
  the PDX builder and to per-list IVF construction.
* `PdxStore` — `Vec<Vec<f32>>` of blocks, each block is `d * bl` (bl == 64
  except last), column-major. Owns a dimension permutation `perm: Vec<u32>`
  chosen at build time (variance-descending by default).

Scans:

* `brute_horizontal(store, q, k)` — reference.
* `brute_pdx(store, q, k)` — walks each block column-by-column, keeps a
  `[f32; 64]` accumulator, no pruning. This is the "layout wins alone" case.
* `brute_pdx_pruned(store, q, k, dim_step)` — after every `dim_step` dims,
  compare each lane's partial distance to the current top-k threshold;
  disable lanes exceeding it; skip disabled lanes on subsequent columns.
* `brute_pdx_block_pruned(store, q, k, dim_step)` — after every `dim_step`
  dims, compare the block's MIN partial distance to the threshold; if it
  exceeds, drop the whole block. Preserves a branchless inner loop.

IVF: `kmeans_assign` builds `n_lists` per-list stores; `probe` picks the
`nprobe` nearest centroids; each probed list is scanned with the chosen
backend and its local top-k is folded into a global heap.

## Implementation notes

* `BLOCK_LEN = 64` is a `const` (not generic). Matches the paper and two
  AVX2 registers of `f32`, one AVX-512 register, or four NEON registers.
* Rust's autovectorizer handles the branchless columnar loop cleanly on
  both AArch64 and x86_64 at `opt-level=3, lto=thin, codegen-units=1`.
  Per-lane `if active[lane]` predicates prevent this — see benchmarks.
* Top-k is a bounded `BinaryHeap` (`TopK`) that doubles as the pruning
  threshold source. `threshold()` returns `+inf` until the heap is full,
  after which it returns the k-th-best distance.
* Dimension permutation is computed once from sample variance and stored
  as `Vec<u32>`. The query is permuted once per scan, not per block.
* Detached workspace (no `[workspace]` inheritance) so the crate is not
  coupled to the top-level `ruvector` build.

## Benchmark methodology

* Dataset: 50 000 vectors, `d = 128`, IID Gaussian (StdRng seed 2026).
* Queries: 100 fresh Gaussian queries (same seed sequence).
* Top-k: `k = 10`.
* IVF: 64 lists, 8 k-means iterations, `dim_step = 8`.
* Warmup: one query per backend before timing.
* Timing: `std::time::Instant`, wall-clock, single thread. Compiled with
  `opt-level = 3, lto = "thin", codegen-units = 1`.
* Recall vs. horizontal brute-force baseline.
* Host: Apple M-series, `cargo 1.77+`, no explicit target-cpu.

## Results

Numbers taken from `cargo run --release --bin pdx-bench` on 2026-09-15
(commit for this branch):

```
== PDX brute-force ==
n=50000, d=128, k=10, queries=100
horizontal bytes ~ 24.4 MiB
PDX bytes         = 24.4 MiB

  horizontal          : 162.43 ms total   1.624 ms/query   1.00x   recall 1.000
  pdx plain           :  55.33 ms total   0.553 ms/query   2.94x   recall 1.000
  pdx pruned (lane)   : 472.24 ms total   4.722 ms/query   0.34x   recall 1.000   work 0.718
  pdx block-prune     :  87.36 ms total   0.874 ms/query   1.86x   recall 1.000   work 0.959

== IVF nprobe study (64 lists) ==
nprobe= 4  candidates ~   3126
  ivf horizontal      : 0.132 ms/query   recall 0.194
  ivf pdx-pruned      : 0.327 ms/query   recall 0.194   work 0.845   speedup 0.40x
nprobe= 8  candidates ~   6259
  ivf horizontal      : 0.235 ms/query   recall 0.320
  ivf pdx-pruned      : 0.634 ms/query   recall 0.320   work 0.846   speedup 0.37x
nprobe=16  candidates ~  12519
  ivf horizontal      : 0.465 ms/query   recall 0.513
  ivf pdx-pruned      : 1.261 ms/query   recall 0.513   work 0.847   speedup 0.37x

== Dimension-order ablation (brute-force PDX pruned, dim_step=8) ==
  identity order      : 4.800 ms/query   work 0.716
  variance-desc       : 4.735 ms/query   work 0.718
```

**Headline numbers:**

| Variant | ms/query | Speedup | Recall |
| --- | --- | --- | --- |
| Horizontal baseline | 1.624 | 1.00× | 1.000 |
| PDX plain (layout only) | **0.553** | **2.94×** | 1.000 |
| PDX branchless block-prune | 0.874 | 1.86× | 1.000 |
| PDX per-lane prune | 4.722 | 0.34× | 1.000 |

## How it works (walkthrough)

Say we have four vectors, `d = 4`. Horizontal storage:

```
[v0.x v0.y v0.z v0.w] [v1.x v1.y v1.z v1.w] [v2.x v2.y v2.z v2.w] [v3.x v3.y v3.z v3.w]
```

Distance kernel: for each vector, load its 4 floats, subtract-square-add
against the query — one reduction per vector.

PDX with block size 4:

```
Block 0:
  dim x : [v0.x v1.x v2.x v3.x]
  dim y : [v0.y v1.y v2.y v3.y]
  dim z : [v0.z v1.z v2.z v3.z]
  dim w : [v0.w v1.w v2.w v3.w]
```

Distance kernel: for each dim, load the 4-lane column, broadcast the
query's dim value, subtract-square, add to four running per-lane
accumulators `acc[0..4]`. After walking all 4 dims, `acc` holds the full
squared distances. No reduction inside the inner loop; the SIMD unit
processes 4 (or 8, 16) lanes at once.

Add pruning: after walking `dim_step` dims (say 2), inspect the four
`acc` lanes. If the current best-of-k threshold is `T`, any lane with
`acc[i] > T` is doomed — we can either mark it dead (per-lane prune) or
drop the whole block once `min(acc) > T` (block prune). Per-lane prune is
"finer" but the resulting branch in the inner loop stops SIMD in its
tracks. Block prune is coarser but stays branchless.

## Practical failure modes

* **`if active[lane]` kills autovec.** Measured 8.5× slowdown vs. plain
  PDX on Gaussian data. The paper hides this behind AVX-512 masked stores;
  in portable Rust the safe way to get lane-level pruning is `std::simd`
  masks (still nightly-only as of 2026-09) or hand-written `core::arch`.
* **IVF lists tighten the threshold slowly.** On k=10 with 3 000–12 000
  candidates per list, pruning kicks in only after most of the list has
  been scanned; block-prune early-exit fires on 94 % of blocks but only
  after ~95 % of dims, so it saves little wall clock.
* **PDX build cost.** Building the PDX store is `O(n * d)` extra work
  vs. horizontal (one transpose per block). Amortize over many queries;
  a build for 1 M × 128 costs ~500 ms and only makes sense for
  read-heavy indexes.
* **Non-Gaussian data.** On clustered data with a heavy tail (e.g. image
  embeddings after normalization), the top-k threshold drops much faster
  and both prune variants can beat plain PDX; the paper's IVF gains come
  from exactly this regime.

## What to improve next

1. **`std::simd` per-lane prune.** Once `core::simd` stabilizes, rewrite
   the per-lane pruned scan with `Simd<f32, 8>` masked stores. Expected
   to recover the paper's advertised 3–4× vs. horizontal.
2. **Fused PDX + PQ.** Store PQ codes in PDX layout (one code column per
   subquantizer). Combined with ADC lookup tables this compresses the
   scan and can compound with SIMD gains.
3. **Threshold-priming.** Seed the top-k heap with distances to a handful
   of centroid-neighbors before scanning, so pruning kicks in from
   dim 0. Rough estimate: 20 % IVF speedup on clustered data.
4. **Hybrid layout.** Keep first-`dim_step` dims horizontal (for the
   threshold-building pass) and the tail in PDX (for the pruning pass).
5. **PDX+HNSW visited-set scan.** During HNSW beam expansion, batch
   candidate neighbors into a temporary PDX block and score with
   `brute_pdx`.

## Production crate layout proposal

```
crates/ruvector-pdx-scan/
├── Cargo.toml           # detached workspace; feature-gate SIMD variants
├── README.md
├── src/
│   ├── lib.rs           # public API
│   ├── block.rs         # Horizontal/PdxStore, VectorStore trait
│   ├── scan.rs          # top-k heap, brute_horizontal, brute_pdx
│   ├── pruned.rs        # brute_pdx_pruned, brute_pdx_block_pruned
│   ├── ivf.rs           # IvfHorizontal, IvfPdxPruned
│   ├── simd/            # future: std::simd variants (nightly-gated)
│   └── bin/pdx_bench.rs # real-numbers benchmark
└── tests/               # currently unit-tests inline; extract for CI
```

Integration path into ruvector proper: expose `PdxStore` as an alternative
backend for `ruvector-cluster` IVF lists; feature-gate behind `pdx-scan`
and default off until per-lane SIMD stabilizes.

## References

1. Kuffo, L.; Krommydas, E.; Boncz, P. **PDX: A Data Layout for Vector
   Similarity Search.** SIGMOD 2024.
2. Gao, J.; Long, C. **High-Dimensional Approximate Nearest Neighbor
   Search: with Reliable and Efficient Distance Comparison Operations.**
   SIGMOD 2023. (ADSampling)
3. Chen, P.-H. et al. **FINGER: Fast Inference for Graph-based Approximate
   Nearest Neighbor Search.** SIGMOD 2023.
4. Malkov, Y. A.; Yashunin, D. A. **Efficient and robust approximate
   nearest neighbor search using Hierarchical Navigable Small World
   graphs.** IEEE TPAMI 2018.
5. Jégou, H.; Douze, M.; Schmid, C. **Product quantization for nearest
   neighbor search.** IEEE TPAMI 2011.
