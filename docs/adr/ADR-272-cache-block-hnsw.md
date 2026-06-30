# ADR-272: Cache-Block Adjacency Layout for HNSW

- **Status**: Proposed (prototyped — branch `research/nightly/2026-06-30-cache-block-hnsw`)
- **Date**: 2026-06-30
- **Crate**: `crates/ruvector-cache-block-hnsw`
- **Research**: [docs/research/nightly/2026-06-30-cache-block-hnsw/](../research/nightly/2026-06-30-cache-block-hnsw/README.md)
- **Related**: ADR-265 (benchmark suite), ADR-267 (SOTA validation), prior HNSW work in `ruvector-coherence-hnsw`, `ruvector-hnsw-repair`.

## Context

HNSW search wall-time at production scale is dominated by L2/L3 cache
misses on graph traversal — confirmed by both our `ruvector-sota-bench`
profiling traces and Lucene's LUCENE-10054 thread. Two of the loads are
inherent (the neighbour vectors themselves), but the **adjacency-list
loads** are structurally avoidable: a textbook CSR layout (`offsets:
Vec<u32>`, `neighbours: Vec<u32>`) splits a single node's list across
two-to-three cache lines and stalls the inner loop.

Adjacent SOTA moves (DiskANN block layouts for SSD, CAGRA warp-coalesced
adjacency for GPUs, Milvus block-packed PQ codes) all converge on the
same principle: **co-locate the data the inner loop needs into the unit
the memory hierarchy delivers it in**. The cache-line analogue is
unexplored in our codebase.

A secondary opportunity: once each node owns a full 64-byte cache line
of metadata, several bytes are free. Spending them on a per-edge
distance proxy enables an *early-reject* before the FP32 distance, which
is the largest single CPU cost per step.

## Decision

Land a new crate `ruvector-cache-block-hnsw` exposing **three swappable
HNSW back-ends** behind a single `AnnIndex` trait:

1. **`BaselineHnsw`** — CSR adjacency. Reference and regression guard.
2. **`BlockHnsw`** — one cache-aligned 64-byte block per node holding 16
   neighbour IDs. Prefetch shim (`_mm_prefetch` on x86_64,
   `prfm pldl1keep` on aarch64) for the first 4 neighbour vectors.
3. **`SketchHnsw`** — same 64-byte block split as 12 IDs + 12 × u8
   norm-bucket sketches. At query time, `|sketch_n − sketch_q| > slack`
   triggers an early-reject without ever loading the neighbour vector.

The trait keeps layout choice swappable per workload — required because
the right answer depends on embedding distribution (unit-normalised vs
not), dataset size relative to L2, and recall/latency budget.

Acceptance bar (all satisfied by the PoC; numbers in the research doc):

- `cargo build --release -p ruvector-cache-block-hnsw` succeeds.
- `cargo test -p ruvector-cache-block-hnsw` — 3/3 pass.
- `cargo run --release --bin cache-block-bench` produces real numbers
  for at least 3 variants on at least 2 dataset sizes.
- BlockHnsw must match BaselineHnsw recall within noise (regression
  guard); the test asserts ≥0.9 overlap, observed 1.00.
- SketchHnsw must expose a slack knob with measurable recall/speed
  trade-off (observed: recall 0.13 → 0.30 → 0.41 across slack
  20 / 64 / ∞, at N=50k).

## Consequences

**Positive**

- Strict-Pareto layout win at the scale where cache traffic matters
  (5.1% faster, 4% less adjacency RAM at N=50k, identical recall).
- Sketch reject opens a coarse-rerank tier (1.5× faster at much lower
  recall) for cascaded retrieval pipelines.
- Trait-based seam lets `ruvector-bench` and `ruvector-cli` swap layouts
  without touching call sites — the layout becomes a build-time choice.
- 64-byte blocks are mmap-friendly: a future on-disk format is a single
  `Vec<AdjBlock>` blob, no per-record header.

**Negative / risks**

- Hard cap `BLOCK_M = 16` (or 12 for sketch). Spillover handling will
  be needed before this can replace the CSR layout in the main HNSW
  crate; tracked as follow-up §3 in the research doc.
- Norm-bucket sketch degenerates to a constant on unit-normalised
  embeddings — most modern text embedding pipelines normalise. For
  those, `BlockHnsw` (no sketch) is the right pick, or the sketch needs
  swapping to a hash-style proxy (follow-up §1).
- Prefetch hint is scalar; needs the prefetch-distance tuned per
  microarch, currently a fixed 4. Easy to make a build-time const.
- Layer-0-only PoC. Upper layers in real HNSW need the same treatment
  before this is a drop-in replacement.

## Alternatives considered

- **Keep CSR, add only vector quantisation** (RaBitQ / LeanVec
  already in the repo): orthogonal; quantisation shrinks the *vector*
  cache pressure but does not address the *adjacency* pointer chase. The
  two stack.
- **GPU warp-coalesced adjacency (CAGRA-style)**: solves the same
  problem on the wrong hardware target for our default CPU path.
- **Block-packed PQ codes only (Milvus 2.4-style)**: same idea but for
  IVF lists rather than graph neighbours; orthogonal index family.
- **JVector-style block decoding without sketches**: equivalent to our
  `BlockHnsw`; we add the sketch variant as a separate, opt-in tier.
- **Move adjacency to mmap'd SSD pages (DiskANN-style)**: solves the
  next memory tier down, not the current one. Worth doing *after* the
  cache-line layout is settled — same block shape, different backing
  store. Tracked as follow-up §6.

## Follow-ups

- §1 superbit-LSH sketches, §2 SIMD block batch, §3 variable-degree
  blocks, §4 upper-layer port, §5 hot-edge reordering, §6 mmap persistent
  format. All enumerated in the research doc.
