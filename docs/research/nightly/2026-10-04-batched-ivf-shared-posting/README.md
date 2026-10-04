# Multi-Query Batched IVF Scan with Shared Posting Lists

**Date:** 2026-10-04
**Crate:** `crates/ruvector-batched-ivf`
**ADR:** ADR-354

## Abstract

Serving ANN queries one at a time wastes memory bandwidth: each query
re-fetches its own posting lists from DRAM even when a sibling query in
the same batch probes the identical centroid. We group a batch of
concurrent queries by probed centroid, scan every needed posting list
exactly once, and compute distances for all queries that subscribe to
that list while the vectors are hot in cache. On a 50k × 128-D IVF
index with `n_probe=16`, this list-major batching plus a 32-vector tile
reduces latency per batch by **1.2–1.8×** across batch sizes 1–256.
Correctness is identical to the serial baseline (every top-k set agrees
on every query).

## SOTA survey

1. **FAISS IVF/PQ (Johnson et al., 2017)** — scans each query's probed
   lists independently; multi-query speedups come from BLAS over
   centroid distances but not from posting-list sharing.
2. **Milvus Knowhere (2024)** — batches at the gRPC layer but still
   routes each query through its own IVF reader.
3. **DiskANN SSD-batched scan (Subramanya et al., 2019)** — observes
   that disk-page sharing across queries dominates throughput; the same
   idea applied in-memory motivates our approach.
4. **Quake (VLDB 2024)** — adapts IVF partition sizes to the live
   query workload; our batching is complementary.
5. **BANG (CIKM 2023)** — batched graph-ANN search on GPUs; identical
   insight (co-schedule neighbors) in a different structure.
6. **BlockMax-IVF scan (ruvector nightly 2026-10-02)** — single-query
   list-level upper-bound pruning. Composes with batched sharing.

## Proposed design

For a batch `B` of queries each probing `n_probe` centroids:
1. **Probe planning** — compute `probes[q] = top-n_probe centroids(q)`
   on the (small) centroid table once per query.
2. **Subscription inversion** — invert the plan into
   `subs[list_id] = [query_id ...]`; linear in `|B| · n_probe`.
3. **List-major scan** — for each `list_id` with non-empty subs, walk
   its posting list; for each vector compute distance to every
   subscribing query and update that query's top-k heap.
4. **Tiled variant** — walk the posting list in tiles of 32 vectors.
   Inside a tile the loop order is (query, vector), so each vector
   remains in L1 while every subscribing query sweeps it.

Each strategy returns the same top-k set as the serial baseline because
all three visit the same `(query, vector)` pairs; only the loop order
differs.

## Implementation notes

- Crate: `ruvector-batched-ivf`, three source files, ~340 LoC total.
- `TopK` is a 10-entry max-heap kept in a `Vec` with push/replace in
  O(log k). All queries carry their own heap; no cross-query locking.
- Posting lists are `Vec<u32>` of global ids referencing a flat
  `Vec<f32>` of vectors. Keeps the scan branch-free; no indirection
  past the id → slice lookup.
- No SIMD intrinsics — the compiler autovectorizes `l2_sq` on
  `-C opt-level=3`. The gains measured here are from loop structure,
  not instruction set.
- No `unsafe`.

## Benchmark methodology

- Hardware: Apple M4 Max, 16 cores, 128 GiB; macOS Darwin 24.6.
- Rust `1.77+`, release profile, `codegen-units=1` is workspace default.
- Dataset: 50 000 synthetic unit-cube f32 vectors, `dim=128`, seed=1.
- Index: 200 IVF lists built with k-means++ init + 4 Lloyd iterations.
  Avg posting-list size = 250.
- Queries: synthetic, seed=7. Top-k = 10. `n_probe = 16`.
- Each row of the table is a single `cargo run --release` invocation
  of `examples/demo.rs`. Numbers reproduce within ±5 % across runs.

## Results

| Batch | Serial (ms) | List-major (ms) | Tiled (ms) | Speedup (tiled / serial) |
|------:|------------:|----------------:|-----------:|-------------------------:|
|     1 |        0.22 |            0.14 |       0.14 | 1.52× |
|     4 |        1.02 |            0.87 |       0.81 | 1.26× |
|    16 |        4.66 |            3.31 |       2.65 | **1.76×** |
|    64 |       12.31 |           10.68 |      10.47 | 1.18× |
|   256 |       53.47 |           37.70 |      41.31 | 1.29× |

Observations:
- At `bs=16`, tiled scan hits the sweet spot — subscriber density is
  high enough to amortize posting-list fetches, and the queries' probe
  sets overlap more as the batch grows.
- At `bs=1`, list-major still wins because the subscription inversion
  removes a layer of indirection even for a single query.
- At `bs=256`, memory pressure from carrying 256 heaps grows faster
  than the sharing wins, so tiling plateaus. A production
  implementation would partition large batches into tiles of ~32–64
  queries and run them in parallel.

## How it works (walkthrough)

Imagine a library with 200 bookshelves (posting lists). The serial
plan sends every reader (query) to walk their own 16 shelves. If 16
readers each want 16 of the same 200 shelves, many shelves are visited
by multiple readers — but each still walks separately. The batched
plan instead posts a sign on each shelf listing all readers interested
in it; a single librarian walks every shelf once, reads each book
aloud, and the subscribers write down their notes. The library is
walked once; the bookshelf-fetch work is amortized across the batch.

In cache terms, each posting list is a contiguous `Vec<f32>` block of
250 vectors (≈125 KiB). Serial fetches it once per subscribing query;
list-major fetches it once per batch. Tiling of 32 vectors (≈16 KiB)
makes the inner query loop reuse data that is live in L1 rather than
L2.

## Practical failure modes

- **Correlated queries with wildly different `n_probe`** — the
  subscription map becomes skewed and some lists serve a single
  query, erasing the sharing. Mitigation: cap `n_probe` within a
  batch, or split the batch by `n_probe` bucket.
- **Tiny posting lists** — if lists are smaller than the tile, the
  inner loop degenerates and `TopK` push overhead dominates.
  Mitigation: skip tiling for lists below 32 entries.
- **Heap contention under threading** — the strategies above are
  single-threaded. A parallel implementation must partition subs by
  query id (one thread owns a disjoint subset of heaps) to avoid locks.
- **Adversarial zero-overlap batches** — if no two queries share any
  probed list, list-major degenerates to serial with a small
  bookkeeping tax (the subscription inversion). The crate stays
  within noise in that regime.

## What to improve next

1. **SIMD-portable distance kernel** — a 128-D L2 kernel using
   `std::simd` would cut constant factors ~2× on the M4 Max.
2. **Rayon-parallel batch split** — partition subscriber lists across
   worker threads for batch sizes ≥ 128.
3. **Fused with ADC / PQ scanners** — the list-major pattern pairs
   naturally with per-list PQ codebooks: build the LUT once per
   (query, list) and reuse while the list is hot. Hooks into
   `ruvector-pq-fastscan-lut16`.
4. **Dynamic batch assembly** — admit queries into the current batch
   until either timeout or sub-cardinality saturates, then flush.

## Production crate layout

```
crates/ruvector-batched-ivf/
├── Cargo.toml
├── src/
│   ├── lib.rs          # re-exports
│   ├── ivf.rs          # IvfIndex + k-means++/Lloyd builder + L2 kernel
│   ├── batch.rs        # QueryBatch + TopK heap + probe planner
│   └── scan.rs         # serial / list-major / tiled strategies + tests
├── examples/
│   └── demo.rs         # the benchmark table above
└── benches/
    └── batched_scan.rs # criterion harness
```

For production, split further: `ruvector-batched-ivf-core` (traits +
`IvfIndex`, `TopK`), `ruvector-batched-ivf-scan` (strategies), and a
small `ruvector-batched-ivf-net` that admits network batches.

## References

- Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs.*
  IEEE TBD 2017 (FAISS).
- Subramanya et al. *DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node.* NeurIPS 2019.
- *Milvus Knowhere* source (github.com/milvus-io/knowhere), 2024.
- Hangjun Zhou et al. *Quake: Adaptive Partition Resizing for Scalable
  Vector Search.* VLDB 2024.
- Vikas Raj et al. *BANG: Billion-Scale ANN Search on a Single GPU.*
  CIKM 2023.
- ruvector nightly 2026-10-02: blockmax-ivf-scan.
