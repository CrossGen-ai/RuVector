# ADR-340: Delta+Varint and PFor-Blocked Adjacency Compression for HNSW Graphs

**Status:** Accepted (research PoC)

**Date:** 2026-08-26
**Owners:** ruvector graph-index maintainers
**Tracking:** `crates/ruvector-delta-varint-hnsw-adjacency` on branch
`research/nightly/2026-08-26-delta-varint-hnsw-adjacency`

## Context

Once vectors are quantized with RaBitQ or PQ-64, the HNSW adjacency
structure — a fixed-M list of `u32` neighbor IDs per node — becomes the
dominant memory cost of the index. At M=32, layer 0 consumes 128 B/node
(4 B ID * 32), which is **larger than the quantized payload** of a
PQ-64 vector (64 B).

For a 100 M-vector corpus this is 12.8 GiB of graph edges alone. This
either forces the index off-DRAM (SPANN, DiskANN mmap) or caps the
supported corpus size on commodity hardware.

Modern IR posting-list compression (delta + varint, PFor with per-block
bitwidth) is a proven solution for the same shape of data — sorted
`u32` ID lists — and has been adopted piecemeal by Milvus (group-varint on
DiskANN) and Qdrant (opt-in LEB128 delta on mmap). No ruvector crate yet
exposes a **swappable, benchmarked, unsafe-free** adjacency backend.

## Decision

Introduce a new workspace crate `ruvector-delta-varint-hnsw-adjacency`
containing:

1. A public trait `AdjacencyStore` with `decode_into(node, out) -> usize`
   as the single hot-path method.
2. Three concrete backends:
   - `PlainAdjacency` — baseline `[u32; M]` slots.
   - `DeltaVarintAdjacency` — sort + delta + LEB128 varint stream, per-node
     offset table.
   - `PforBlockedAdjacency` — sort + delta + per-node fixed-bitwidth
     packing with a 6-byte header (`[len:u8][bw:u8][id0:u32]`).
3. A deterministic `synth_neighbors` generator and a real `cargo bench`
   harness that emits CSV of memory footprint and decode ns/op.

`#![forbid(unsafe_code)]`. Zero external dependencies.

The crate is a research PoC deliverable — it is not yet wired into any
existing HNSW crate. Integration is the next ADR after SIMD unpack lands
(see Consequences → What's next).

## Consequences

**Positive**

- 2.0×–3.1× reduction in adjacency memory depending on graph locality,
  reproducible on M4 Max in-tree (`cargo bench -p ruvector-delta-varint-hnsw-adjacency`).
- Trait-based design lets `ruvector-coherence-hnsw`, `ruvector-diskann`,
  and `ruvector-spann` opt in per index without a source rewrite.
- Neither `unsafe` nor a new dependency is introduced into the workspace.
- Lossless — proven by 6 roundtrip tests including a compression-ratio
  invariant.

**Negative / accepted costs**

- Decode is 5–7× slower per node than memcpy (13 → 70–130 ns on M4 Max at
  M=32). For an `ef=64` search this adds < 90 µs per query — negligible
  next to distance compute.
- Neighbor *order* is discarded (sorted at build time). HNSW search does
  not need it, but any variant that piggybacks a stored distance on
  neighbor index (e.g. adaptive-recall's per-edge weight) needs a
  parallel payload array.
- Write path is not amortized: an insert re-encodes an entire node row.
  On write-heavy workloads, callers should keep `PlainAdjacency` during
  the hot window and compact to PFor on snapshot.

**What's next (deferred to future ADRs)**

- SIMD (NEON / AVX-512) bit-unpack — expected 2–3× decode speedup.
- Per-64-element PFor with exceptions (classic Zukowski) for high-M graphs.
- `AdjacencyStore::prefetch(node)` hook so HNSW greedy search can overlap
  decode with the next candidate's cache miss.
- Graph-relabelling (recursive bisection / Hilbert) pre-pass, which is
  what closes the gap between "uniform" (2×) and "loc=0.98" (3×).

## Alternatives considered

- **Group-varint (Milvus).** Simpler decode, but ~10 % worse ratio than
  LEB128 at this M and no per-node bitwidth adaptivity. Kept in the
  roadmap as a possible future backend for the "DiskANN long-tail" case.
- **Roaring bitmaps.** Optimized for very large sparse sets, not for M=16-64
  fixed-size lists. Overhead of the container header dominates.
- **Elias–Fano.** Optimal for asymptotically-large monotone lists; for M=32
  the constant overhead makes it slower and larger than PFor.
- **On-disk only (mmap + delta-varint).** What Qdrant does. Leaves in-RAM
  wins on the table — this ADR wants the wins even for in-memory graphs.

## References

- Research doc: `docs/research/nightly/2026-08-26-delta-varint-hnsw-adjacency/README.md`
- Crate: `crates/ruvector-delta-varint-hnsw-adjacency/`
- Bench: `cargo bench -p ruvector-delta-varint-hnsw-adjacency`
- ADR-322 (matryoshka-coarse-fine) and ADR-336 (reversible signed transactions)
  are the closest prior art in the ruvector ADR series on quantized index
  layouts.
