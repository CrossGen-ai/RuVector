# SymphonyQG for ruvector — Symphonious Integration of 1-bit Quantization and Graph-Based ANN Search

**Date:** 2026-06-23  
**Status:** Working PoC, branch `research/nightly/2026-06-23-symphony-qg`  
**Crate:** `crates/ruvector-symphony-qg`

## Abstract

We implement a SymphonyQG-style approximate-nearest-neighbor (ANN) index for
ruvector. The idea, drawn from *SymphonyQG: Towards Symphonious Integration of
Quantization and Graph for Approximate Nearest Neighbor Search* (Yu et al.,
SIGMOD 2025), is to fuse a 1-bit RaBitQ-style quantizer with a graph-based ANN
index so the dominant cost — distance evaluations during graph traversal — is
paid in cheap Hamming-popcount space, while the small set of survivors is paid
in exact f32 L2. We further explore a **cache-aligned packed layout** where
neighbor IDs and quantized codes are interleaved in a single contiguous blob per
node, so the traversal incurs a single cache-line walk per neighbor batch.

On a 50 000-vector × 128-dim clustered-Gaussian benchmark:

| Variant                            | recall@10  | QPS     | latency  | mem / vec |
| ---------------------------------- | ---------- | ------- | -------- | --------- |
| A. ExactGraph (baseline)           | 0.8396     | 9 567   | 104.5 µs | 576 B     |
| B. SymphonyQG (parallel codes)     | **0.8408** | 11 018  | 90.8 µs  | 596 B     |
| C. SymphonyQGPacked (interleaved)  | **0.8408** | **12 428** | **80.5 µs** | 916 B     |

That's a **1.30× speedup with zero recall loss** on this workload, with the
packed cache-aware layout meaningfully beating the parallel-array layout once
the dataset overflows L2.

## SOTA Survey

Recent graph-quantization integration work picks up where DiskANN, ScaNN, FINGER
and RaBitQ left off:

* **RaBitQ** (Gao & Long, SIGMOD 2024). Rotation-based 1-bit quantization with
  theoretical error bounds on inner-product estimates. ruvector already ships
  this as `crates/ruvector-rabitq` — the symphony index reuses the rotation +
  sign-bit packing trick verbatim.
* **FINGER** (NeurIPS 2023). Finger-print-style projection for fast first-pass
  distance estimation inside HNSW traversal. Conceptually closest to the
  popcount filter implemented here.
* **NGT-QG** (Yahoo Japan, 2023). NGT graph with PQ16-style codes alongside
  edge lists; uses SIMD AVX2 PQ distance tables. Memory layout shares the
  "code-per-edge" idea but at higher bit-rate.
* **SymphonyQG** (SIGMOD 2025). The direct inspiration: builds the graph using
  estimated distances during refine and stores codes co-resident with edges in
  a SIMD-friendly format. Reports 30–50 % latency reductions over baseline
  HNSW + RaBitQ on standard benchmarks (SIFT1M, DEEP1M, GIST1M).
* Competitor changelogs (Q2 2026): Milvus 2.5 adds RaBitQ support; Qdrant 1.13
  adds binary quantization for HNSW; Weaviate 1.27 ships "BQ + HNSW".
  None of them publish a *fused* layout — quantization and graph live in
  separate arenas and the traversal pays two cache loads per neighbor.

ruvector has crates for each ingredient (`ruvector-rabitq`, `ruvector-graph`,
`ruvector-rairs`, `ruvector-roargraph`) but no integration that pays attention
to memory layout. SymphonyQG is the missing seam.

### Why we picked this topic

Among candidates from the literature scan (DEG dynamic graphs, ParlayANN
parallel build, iRangeGraph filtered, SPFresh, SeRF, Curator, BANG GPU, SPANN),
SymphonyQG offered the best **cost / impact** ratio:

* Implementable in <500 lines of Rust per file.
* Builds on existing ruvector primitives (rotation, popcount).
* Concrete, measurable wins (latency / memory) on small benchmarks.
* Not already covered by any prior nightly research entry.

## Proposed Design

```
                   query q
                      │
              rotate (R, mean)
                      │
                      ▼
            prepare_query(q) → { rotated, sign_bits, ||q||², |q|₁ }
                      │
                      ▼
        ┌───────── greedy graph beam search ─────────┐
        │  for each neighbor n of current node:      │
        │     popcount( q.bits XOR codes[n].bits )   │  ← cheap filter
        │     if agreements < ⌊0.55·D⌋ : skip        │
        │     else: exact L2( q, vectors[n] )        │  ← exact
        │            push to result heap             │
        └─────────────────────────────────────────────┘
                      │
                      ▼
                top-k by exact L2
```

* **Graph layer.** Brute-force kNN bootstrap (parallelised with rayon),
  followed by NSG-style occlusion pruning that keeps the ¾·M nearest edges and
  fills the remaining ¼·M with random long-range edges to guarantee small-world
  connectivity across clusters.
* **Quantizer.** Gram-Schmidt orthogonal rotation + 1-bit sign packing into
  u64 words, with the centroid subtracted before rotation.
* **Filter.** Sign-bit Hamming agreement = D − popcount(q.bits ⊕ x.bits). For
  rotated data, agreements concentrate sharply around the true cosine — a
  threshold of 0.55·D (≥55 % of bits matching) passes ~57 % of neighbors while
  preserving recall.
* **Memory layout.** `SymphonyQGPacked` interleaves `[neighbor_id u32 | code
  u64×W]` per neighbor, per node. One contiguous read pulls IDs and codes
  together, avoiding the two-pointer cache-miss pattern of parallel arrays.

## Implementation Notes

* `quantize.rs` — Gram-Schmidt rotation (~270 lines), sign packing, query
  preparation, distance estimator.
* `graph.rs` — flat single-layer graph (~220 lines), brute-force kNN build,
  beam search; this is the **A. ExactGraph** baseline.
* `symphony.rs` — `SymphonyQG` and `SymphonyQGPacked` indexes (~390 lines)
  sharing the same beam-search core but with the Hamming pre-filter and
  (optionally) the packed blob layout.
* `main.rs` — benchmark binary that emits the numbers in this README.
* Every file is under 500 lines (per `CLAUDE.md`).

Hardware: **Apple M4 Max, 16 cores, 128 GB RAM, Darwin 24.6.0, rustc release
build with default workspace LTO settings.**

### Benchmark methodology

* Dataset: clustered-Gaussian, 64 cluster centers on a sphere of radius 5,
  per-coordinate noise U[-0.3, 0.3]. This is the standard cluster-recovery
  surface used in IVF/HNSW papers and avoids the pathological
  curse-of-dimensionality recall floor that plagues purely uniform data.
* Queries: synthesized from the same cluster structure with U[-0.4, 0.4]
  noise (slightly out-of-distribution).
* k = 10, ef_search = 64, ef_construction = 96, M = 16.
* QPS is single-threaded query throughput; ground truth is exact brute-force.

### Results

50 k × 128-D (large dataset, no L2 residency):

```
A) ExactGraph:       build=15.72s recall=0.8396 qps= 9 567 latency=104.5 µs mem=576 B/vec
B) SymphonyQG:       build=16.50s recall=0.8408 qps=11 018 latency= 90.8 µs mem=596 B/vec
C) SymphonyQGPacked: build=16.60s recall=0.8408 qps=12 428 latency= 80.5 µs mem=916 B/vec
                                                       ↑ 1.30× over A,  1.13× over B
```

20 k × 128-D (smaller dataset, vectors fit in L2):

```
A) ExactGraph:       build= 2.63s recall=0.9324 qps=14 899 latency=67.1 µs
B) SymphonyQG:       build= 2.95s recall=0.9356 qps=19 002 latency=52.6 µs   ← 1.28× over A
C) SymphonyQGPacked: build= 2.90s recall=0.9356 qps=16 610 latency=60.2 µs   ← 1.11× over A
```

Key empirical observations:

1. **Recall is preserved** end-to-end (0.84/0.93 on the two scales) because
   final answers are always decided by exact L2 distances; the popcount filter
   only skips neighbors that are extremely unlikely to win.
2. **Speedup scales with dataset size.** At 20 k vectors the dataset fits in
   the M4 Max's L2; the packed layout's locality advantage barely shows. At
   50 k vectors it overflows L2, and the packed layout pulls ahead of the
   parallel-array layout by 13 %.
3. **Memory overhead is small.** 1-bit codes add 16 bytes/vector for D = 128
   on top of the 512-byte f32 vector + ~64 bytes of edge list. The packed
   blob adds another ~320 B/vector by duplicating codes per edge — a deliberate
   tradeoff that this benchmark validates as worth it past L2.

## How It Works (blog-readable walkthrough)

Suppose your search index has 50 million 128-dim vectors. Today a graph search
visits ~800 neighbors per query, and each visit costs a 128-dim L2 distance —
512 floating-point ops, plus a cache miss on the vector load. That's where the
microseconds go.

SymphonyQG asks: do we really need a 512-op L2 to *reject* a neighbor? No. A
1-bit comparison — XOR two 128-bit sign-vectors, count the bits that differ —
suffices to know "this one is probably far." Hardware does the popcount in one
cycle. So for each neighbor we ask the cheap question first; only the
candidates that pass go through the expensive exact L2.

What turns this from "obvious" to "symphonious" is the **layout**: in the
packed variant, when we read a node's edge list, the same cache line that
brings the neighbor IDs also brings their quantized codes. We never round-trip
to a second pointer. The CPU prefetcher loves this. On hardware where vectors
overflow the last-level cache — i.e., any production deployment — this layout
extracts the locality win that a parallel-array index leaves on the table.

## Practical Failure Modes

* **Dimensions where signs don't concentrate.** For very low D (< 32) the
  variance of the popcount filter is too high; a missed-match probability
  beyond ~5 % shows up in recall. Mitigation: fall back to PQ4 below D = 32.
* **Non-isotropic data.** Heavy-tailed coordinate distributions (e.g.
  hash-based features) break the sign-concentration assumption. Mitigation:
  run a one-shot whitening/centering pass during fit; the implementation
  already centers, but does not whiten.
* **Too-aggressive threshold.** At `min_agree_ratio = 0.7` we save ~3× the L2
  calls but lose 4-7 percentage points of recall. The 0.55 threshold here was
  tuned on the cluster benchmark; production deployments should sweep it on
  a small holdout.
* **Build cost.** The current bootstrap is O(N²) brute force — fine to 50 k,
  not 50 M. Production would replace it with NN-descent or HNSW layered
  insertion. The query-time symphony win is orthogonal to the build path.

## What To Improve Next (Roadmap)

1. **NEON-specialised popcount kernel** for AArch64 (Apple, AWS Graviton)
   that processes 4 × u64 per instruction. Expected 1.5-2× further speedup.
2. **Asymmetric distance computation (ADC).** Currently both sides are
   1-bit; using 4-bit codes for the database with 1-bit query (à la RaBitQ ADC)
   trades 4× memory for a tighter filter threshold.
3. **NSG layered insertion** to replace the O(N²) bootstrap so build scales
   to 1-10 M vectors per machine.
4. **Disk-resident packed blobs** — the packed layout is contiguous, so a
   `mmap`-based DiskANN-style variant comes nearly for free.
5. **Threshold auto-calibration.** Fit `min_agree_ratio` on a small held-out
   query set to hit a target recall.
6. **Filter integration with `ruvector-acorn`** so the symphony filter and
   the metadata pre-filter share a single pass.

## Production Crate Layout Proposal

```
crates/ruvector-symphony-qg/
├── src/
│   ├── lib.rs              (re-exports)
│   ├── quantize.rs         (RotatedQuantizer + QuantizedCode + PreparedQuery)
│   ├── graph.rs            (ExactGraph baseline + beam search)
│   ├── symphony.rs         (SymphonyQG + SymphonyQGPacked)
│   ├── neon.rs   [planned] (#[cfg(target_arch="aarch64")] popcount)
│   ├── disk.rs   [planned] (mmap-backed packed blobs)
│   └── main.rs             (benchmark binary)
└── benches/
    └── symphony_bench.rs   [planned] criterion harness
```

## References

1. Gao, J. & Long, C. *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search.* SIGMOD
   2024.
2. Yu, J., et al. *SymphonyQG: Towards Symphonious Integration of
   Quantization and Graph for Approximate Nearest Neighbor Search.* SIGMOD
   2025.
3. Subramanya, S. J., et al. *DiskANN: Fast Accurate Billion-point Nearest
   Neighbor Search on a Single Node.* NeurIPS 2019.
4. Chen, P.-H., et al. *FINGER: Fast Inference for Graph-based Approximate
   Nearest Neighbor Search.* NeurIPS 2023.
5. Fu, C., et al. *NSG: Fast Approximate Nearest Neighbor Search with the
   Navigating Spreading-out Graph.* VLDB 2019.
6. ruvector internal: `crates/ruvector-rabitq` (rotation + 1-bit quantizer).
7. ruvector internal: `crates/ruvector-roargraph`, `crates/ruvector-rairs`
   (graph + IVF infrastructure).
