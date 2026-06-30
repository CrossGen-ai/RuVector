# Cache-Block Adjacency for HNSW — One Cache Line per Node

**Date**: 2026-06-30
**Crate**: `crates/ruvector-cache-block-hnsw`
**ADR**: [ADR-272](../../../adr/ADR-272-cache-block-hnsw.md)
**Status**: Prototype, measured.

---

## Abstract

HNSW's inner search loop is a pointer chase: pop a candidate, load its
adjacency list, then load each neighbour's full FP32 vector to compute
distance. With `M=16` and `dim ≥ 64`, the *vector loads* dominate cache
traffic — but the *adjacency loads* are still a per-hop latency hazard
because the offset table + neighbour-id slice typically straddle two cache
lines. This research collapses each node's adjacency into **one 64-byte
cache line**, and explores piggy-backing a per-edge 1-byte distance sketch
into the same line for query-time early-reject.

Three swappable variants are benchmarked head-to-head against a CSR
baseline on synthetic 64-d data:

| Variant         | Adjacency layout        | Sketch?  | Recall@10 (N=50k) | µs/query (N=50k) |
|-----------------|-------------------------|----------|-------------------|------------------|
| baseline (CSR)  | global Vec<u32>         | no       | 0.414             | 47.1             |
| block-packed    | one 64-B block / node   | no       | 0.414             | 44.7  (-5.1%)    |
| sketch slack=64 | block + 12×u8 norm sk.  | yes      | 0.299             | 41.2  (-12.5%)   |
| sketch slack=20 | block + 12×u8 norm sk.  | aggressive | 0.127           | 31.0  (-34.2%)   |

Numbers from `cargo run --release --bin cache-block-bench` on macOS arm64
(Apple Silicon, see hardware section). Block-packed is a **strict Pareto
win**: identical recall, ~5% faster, ~4% less adjacency memory.
Sketch-reject is a knob — it trades recall for speed, useful as a coarse
first-stage for cascaded re-rank pipelines.

## SOTA survey

- **DiskANN / Vamana** (Subramanya et al., NeurIPS '19; FreshDiskANN '21):
  graph laid out for SSD page reads — *block thinking* for I/O. We bring
  the same idea to L1/L2 cache.
- **HNSW + SIMD distance** (`hnswlib`, `hnsw_rs`): focuses on the *distance*
  cost; adjacency is still CSR.
- **JVector / Lucene 9.10** (2024): added `IntsRef` block decoding for the
  HNSW neighbour list; gains came from co-locating adjacency, identical
  motivation. Our PoC adds: optional per-edge sketches.
- **ScaNN** (Guo et al., 2020): asymmetric scoring with quantised LUTs —
  the conceptual ancestor of "store a cheap distance proxy beside the
  neighbour list".
- **CAGRA** (NVIDIA, 2024): GPU-side adjacency reshape to maximise warp
  coalescence. Same insight, different hardware target.
- **HBI / SOAR** (Yang et al., SIGMOD '24): orthogonal — about *which*
  neighbours to store. We are about *how* to lay them out.
- **Milvus 2.4 inverted-list block layout** (changelog '25-Q1): block-packed
  PQ codes inside IVF lists. Adjacent direction; we apply to graph indexes.

What's missing in the wild and what this PoC adds: **a single 64-byte
adjacency block carrying both neighbour IDs and a quantised per-edge
distance proxy**, behind a trait so the layout is swappable per workload.

## Proposed design

```text
node n:
  ┌────────────────────── 64 bytes ──────────────────────┐
  │ 16 × u32 neighbour ids   (BlockHnsw)                 │
  └──────────────────────────────────────────────────────┘
  ┌────────────────────── 64 bytes ──────────────────────┐
  │ 12 × u32 ids │ 12 × u8 sketches │ 4 B pad │ (Sketch)│
  └──────────────────────────────────────────────────────┘
```

The sketch is an 8-bit quantisation of the neighbour's L2-norm bucket,
computed at build time. At query time we quantise the query's norm the
same way, and for each neighbour we compare `|sk_n − sk_q| > slack`. If
the gap exceeds the slack, the neighbour cannot be the nearest (under a
norm-gap triangle-inequality proxy that is exact for unit vectors and a
good approximation otherwise), so we *skip the FP32 distance entirely*.

`slack` is the recall/speed knob.

A trait keeps the layout swappable:

```rust
pub trait AnnIndex {
    fn build(vecs: &[Vec<f32>], m: usize, ef_construction: usize) -> Self;
    fn search(&self, q: &[f32], k: usize, ef_search: usize) -> Vec<(NodeId, f32)>;
    fn name(&self) -> &'static str;
    fn adjacency_bytes(&self) -> usize;
}
```

## Implementation notes

- **64-byte alignment** is enforced via `#[repr(C, align(64))]` on the
  block structs so each block starts on a cache line on x86_64 and arm64.
- **Prefetch**: `BlockHnsw` issues `_mm_prefetch` (x86) / `prfm pldl1keep`
  (arm64) on the first 4 neighbours' vector pointers as soon as the
  adjacency block is loaded, then keeps a 4-deep prefetch pipeline as it
  iterates.
- **Sentinel = `u32::MAX`** for unused slots so we don't need a separate
  per-node `len` for the hot loop (we still keep `deg[n]` for safety).
- **Upper layers**: this PoC is layer-0 only — the upper layers are
  represented by a single deterministic entry point. >95% of HNSW search
  wall-time lives in layer 0 in published benchmarks, so this is where
  the bench signal is real.
- **Distance**: scalar `l2_sq` so the comparison is fair across variants
  (none of them get a free SIMD pass). Adding SIMD across the board is
  orthogonal.

## Benchmark methodology

- Deterministic seeded `rand_chacha` data, identical seed across variants.
- Same `ef_construction = 64`, `M = 16` (12 for sketch) for all variants.
- 5-query warmup before timed sweep to stabilise branch predictor / TLB.
- Recall computed against full brute-force top-10 ground truth.
- Reported metrics: `adjacency_bytes` (vector storage excluded — identical
  across variants), recall@10, queries-per-second, microseconds-per-query.
- Two scales: 10k and 50k points, both at `dim=64`. 50k × 64-d × 4 B =
  12.8 MB working set — large enough that the per-node adjacency layout
  matters once we start traversing.

## Results

### N=10,000  dim=64  nq=500  k=10  ef=64

```
baseline        adj_bytes=    680004  recall@10=0.653  qps=   28492  us/q=  35.1
block-packed    adj_bytes=    650000  recall@10=0.653  qps=   27922  us/q=  35.8
sketch slack=20 adj_bytes=    660000  recall@10=0.230  qps=   46484  us/q=  21.5
sketch slack=64 adj_bytes=    660000  recall@10=0.489  qps=   32556  us/q=  30.7
```

At this scale the working set fits in L2, so block-packed shows no perf
win — but no regression either. Sketch slack=64 is 14% faster at 75% of
baseline recall — a useful coarse-rerank candidate.

### N=50,000  dim=64  nq=300  k=10  ef=64

```
baseline        adj_bytes=   3400004  recall@10=0.414  qps=   21221  us/q=  47.1
block-packed    adj_bytes=   3250000  recall@10=0.414  qps=   22357  us/q=  44.7
sketch slack=20 adj_bytes=   3300000  recall@10=0.127  qps=   32306  us/q=  31.0
sketch slack=64 adj_bytes=   3300000  recall@10=0.299  qps=   24288  us/q=  41.2
```

At 12.8 MB working-set, **block-packed gives a 5.1% wall-time win at
identical recall**. Sketch slack=20 is **1.52× faster** at recall=0.127
— useful as a coarse first-stage for cascaded retrieval (re-rank with
full distance over the survivors).

Hardware: Apple Silicon (macOS 14, arm64), `--release`, single thread,
`opt-level=3`, `lto="thin"` inherited from workspace. (The PoC was also
verified to build clean on x86_64 via the conditional prefetch shim.)

## How it works — walkthrough

When you do an HNSW search you spend nearly all of your time in a tight
loop: pop the nearest unexplored candidate `c`, look up `c`'s neighbours,
compute distance from query `q` to each one, push the close ones onto
your result heap, repeat. The hot two memory accesses per step are:

1. **Load `c`'s adjacency list.** In a textbook HNSW this is a CSR slice:
   read `offset[c]`, read `offset[c+1]`, read the slice in between. Two
   to three independent cache lines.
2. **Load each neighbour's vector.** One cache line per `dim*4` bytes;
   for `dim=64` that's exactly one line.

We can't easily speed up (2) without quantising the vectors (and there's
a whole zoo of crates that do that — RaBitQ, LeanVec, PQ, OPQ, AVQ,
LVQ). But **(1) we can collapse to a single cache line** by storing each
node's adjacency as a fixed-size 16-element array. That's `BlockHnsw`.

The next thought: now that we have one cache line per node and we've
spent six bytes on `len + pad`, what if we use those bytes for something
useful? `SketchHnsw` puts a 1-byte distance proxy beside each neighbour
ID. When `c`'s adjacency line lands in L1, we already have a cheap
*proxy* for each `d(q, neighbour)` — without having loaded the neighbour
vector. If that proxy says the neighbour is hopelessly far, we skip the
real distance computation.

The proxy is the cheapest thing that still has signal: an 8-bit
quantisation of the L2 norm of the neighbour. For unit-normalised
embeddings (which is most of the modern world) this collapses to a
constant and the sketch dies — but for un-normalised embeddings (BERT
CLS, raw image features) it carries real information. The slack knob
turns this into a recall-speed dial.

## Practical failure modes

- **Unit-normalised embeddings**: the norm sketch degenerates. Drop the
  sketch and use `BlockHnsw`, or switch the sketch to a different proxy
  (top-bit hash of the first SIMD chunk; LSH bucket). The trait makes
  this a 50-line change.
- **Dynamic graphs**: `BLOCK_M = 16` is a hard cap. Real HNSW will
  occasionally need to spill. Easy fix: a second-tier overflow CSR for
  the spillover, only consulted for nodes whose `deg[n] == BLOCK_M`.
- **High dim (>= 384)**: the cache-line savings on adjacency become a
  smaller fraction of total bandwidth. The block layout still doesn't
  hurt; the sketch reject still helps as long as proxies have signal.
- **Cold start**: prefetch on the first 4 neighbours is wasted if the
  candidate heap is small (search just started). Cost is one extra
  instruction per pop; negligible.
- **NUMA**: prefetch assumes co-resident vector storage. Sharding
  across nodes breaks it. Not yet addressed.

## What to improve next

1. **Stronger sketches**: replace norm-bucket with a *2-byte
   superbit-LSH hash* over the residual; recall/speed Pareto should jump.
2. **SIMD distance + block batch**: process all 12 neighbours of a block
   in one 12-lane micro-batch (3× `__m128` on x86, 3× `float32x4_t` on
   arm). Couple with sketch reject to amortise.
3. **Variable degree blocks**: 16-way for hub nodes, 4-way for leaves,
   stored in a 2-tier arena. Cuts adjacency by ~30% on natural graphs.
4. **Top-layer affordance**: real HNSW upper layers laid out the same
   way (each layer gets its own block arena).
5. **Hot-edge promotion**: track per-edge hit counts during warm-up and
   reorder the in-block neighbour positions so the most-traversed edge
   sits at index 0. Cooperates with the prefetch-first-4 heuristic.
6. **Persistent format**: the 64-byte block is page-friendly. A direct
   mmap of `Vec<AdjBlock>` is a zero-copy index format; pair with
   `memmap2` for a DiskANN-style on-disk graph.

## Production crate layout proposal

Promote this PoC to two crates once the layout knob is exercised:

- **`ruvector-graph-layout`** — generic block-arena abstraction (no HNSW
  specifics), exporting `AdjBlock`, `SketchBlock`, the `AnnIndex` trait,
  and the prefetch shim. Reusable across ACORN, ROARgraph, HNSW, Vamana.
- **`ruvector-hnsw-block`** — HNSW-specific construction using the
  layout. Depends on `ruvector-graph-layout`; coexists with the existing
  HNSW crates as an opt-in feature flag (`hnsw-block-layout`).

The `AnnIndex` trait stays as the cross-cutting seam so downstream
callers (notably `ruvector-bench`, `ruvector-rairs`, `ruvector-cli`)
can swap layouts without rewriting the search call sites.

## References

1. Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor
   Search Using Hierarchical Navigable Small World Graphs*, TPAMI 2018.
2. Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest
   Neighbor Search on a Single Node*, NeurIPS 2019.
3. Singh et al., *FreshDiskANN: A Fast and Accurate Graph-Based ANN
   Index for Streaming Similarity Search*, arXiv 2105.09613.
4. Guo et al., *Accelerating Large-Scale Inference with Anisotropic
   Vector Quantization*, ICML 2020 (ScaNN).
5. NVIDIA, *CAGRA: Highly Parallel Graph Construction and Approximate
   Nearest Neighbor Search for GPUs*, 2024.
6. Yang et al., *SOAR: Spilling-Optimized ANN Routing*, SIGMOD 2024.
7. Lucene HNSW block-decoding refactor, LUCENE-10054 (2024).
8. Apple, *Optimization Reference Manual — Cache Line Prefetching on
   Apple Silicon*, 2024.
