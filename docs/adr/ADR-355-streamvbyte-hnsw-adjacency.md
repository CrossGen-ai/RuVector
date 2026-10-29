# ADR-355: StreamVByte delta encoding for HNSW adjacency

- **Status**: Proposed
- **Date**: 2026-10-29
- **Deciders**: RuVector Architecture Team
- **Related**: ADR on HNSW storage (`hnsw-delta-neighbors`, `elias-fano-hnsw-adjacency` research); nightly 2026-10-29
- **Tags**: hnsw, graph-storage, compression, simd, cache-locality

## Context

HNSW adjacency lists dominate the RAM footprint of a graph-based ANN
index at mid-to-large degree (M=16–64). Today several ruvector HNSW
backends store adjacency as `Vec<Vec<u32>>` or a packed `[M]u32` block,
burning 4 bytes per neighbor id regardless of actual id magnitude. Prior
nightlies have explored plain delta (`hnsw-delta-neighbors`) and bit-packed
Elias–Fano (`elias-fano-hnsw-adjacency`) encodings. A third codec class —
byte-oriented delta codecs with separated control streams, specifically
**StreamVByte** (Lemire 2017) — has not been evaluated against our workload
and is a known fast middle-ground on x86 servers via SSSE3 shuffle.

Measured on this host (Apple M1 Pro, scalar) across N=10k/100k/500k graphs:

- Plain u32: 4.19–4.38 bytes/edge.
- Delta + LEB128 varint: 2.15–2.52 bytes/edge, ~1.7–2.0× smaller than plain.
- Scalar StreamVByte-delta: 2.29–2.43 bytes/edge, ~1.7–1.9× smaller.

Decode throughput: varint and StreamVByte are within ~1.6× of plain and
~30% of plain on the largest graph. SSSE3-vectorized StreamVByte on x86
is expected (per the paper) to decode 2–3× faster than scalar varint.

## Decision

1. **Introduce `ruvector-streamvbyte-hnsw` as an experimental codec crate**
   with an `AdjacencyCodec` trait and three implementations: `PlainCodec`,
   `DeltaVarintCodec`, `StreamVByteCodec`. A `CompressedAdjacency<C>`
   wrapper provides columnar `(offsets, lens, data)` storage keyed by
   node id.
2. **Keep the trait additive.** HNSW builders in `ruvector-core` are not
   changed in this ADR; integration lands in a follow-up once a SIMD
   decode path is in place and a real-workload benchmark replaces the
   synthetic one here.
3. **Target StreamVByte for the production path**, not plain varint,
   because (a) it compresses within 1% of varint at scale, (b) its decode
   path has a well-known SSSE3 implementation that is faster than any
   varint variant on x86, and (c) its control-stream layout is friendlier
   to prefetching and vectorization.
4. **Preserve varint and plain in the crate as baselines**. They are
   cheap to maintain and essential for differential testing of the SIMD
   path when it lands.
5. **Codec choice is per-layer, not global.** Future HNSW integration
   will let callers pick: dense low-id lists on L0 may still prefer
   Elias–Fano (`elias-fano-hnsw-adjacency`), while sparser upper layers
   favor StreamVByte.

## Consequences

**Positive**

- Immediate 1.7–2.0× reduction in HNSW adjacency RAM at acceptable
  decode cost (demonstrated numbers in the research doc).
- A clean seam (`AdjacencyCodec` trait) that lets us evolve codecs
  without re-shaping the HNSW graph layer.
- Measurable baseline for the SIMD follow-up (ADR-TBD).

**Negative**

- Mutation cost. StreamVByte is a batch codec. In-place neighbor edits
  require decode → patch → re-encode of the full list. Fine for the
  current HNSW build pipeline (neighbor-selection rewrites the whole
  list anyway) but a mutable-graph codepath will need an uncompressed
  overlay or tombstone log.
- Added API surface. We now have three HNSW adjacency codecs across the
  codebase: plain, delta/EF, and StreamVByte. Per-layer choice must be
  documented clearly to avoid confusion.
- Scalar decode is competitive, not dominant, on Apple silicon. The
  real win needs the SIMD path.

## Alternatives considered

1. **Keep plain u32 adjacency.** Simplest, but gives up the ~2× RAM win.
   For N=500k×32 graphs that's ~28 MB per index — meaningful on edge
   hardware and keeps the whole graph out of a higher cache tier.
2. **Standardize on Elias–Fano** (per `elias-fano-hnsw-adjacency`
   research). Better compression on very dense delta distributions,
   worse on sparser ones; decode is bit-level and does not SIMD as
   cleanly as StreamVByte's shuffle. Best as a per-layer peer, not a
   universal replacement.
3. **FastPFor / SIMD-BP128.** Lower bytes/edge than StreamVByte on
   large blocks but needs 128+-element blocks to amortize headers,
   which doesn't fit per-node adjacency (M=32).
4. **Compress the whole graph with zstd** (DiskANN style). Great on
   disk; useless at query time because random-access decode is not
   supported — would require per-node decoding to a scratch buffer on
   every traversal step, same shape as what we're building here but
   with a slower codec.
