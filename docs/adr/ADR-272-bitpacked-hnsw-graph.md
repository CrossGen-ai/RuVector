# ADR-272: Bit-Packed HNSW Neighbor Lists

- **Status**: Proposed (PoC in `crates/ruvector-bitpacked-hnsw`, PR TBD)
- **Date**: 2026-07-07
- **Related**: ADR on `ruvector-diskann`, ADR-268 (SPANN partition-spill), any prior
  graph-layout ADR for `ruvector-core` HNSW.

---

## Context

HNSW graph adjacency accounts for the majority of the in-memory footprint of
RuVector's graph indexes at typical parameters. hnswlib/FAISS store neighbor
lists as `Vec<u32>` (4 bytes/edge, plus a length prefix). At `N = 10 M`,
`M = 32` this is ~1.28 GiB of graph — roughly the same size as a fp16 embedding
table for `d = 32`. Vector quantisation research (RaBitQ, PQ, LVQ) has
attacked the *vector* side of the memory bill; the *graph* side is untouched
in every RuVector crate today.

Neighbor lists are highly compressible. After sorting, consecutive-id deltas
are small (median ≈ N/(M·2), heavily skewed toward zero). Inverted-index
compression research (Anh–Moffat binary-packed, Lemire SIMD-BP128, Boldi–Vigna
WebGraph) has shown 2–5× compression on similar data for two decades, but no
mainstream HNSW implementation applies it.

## Decision

Introduce a `NeighborStore` trait and three swappable backends in a new
research crate `ruvector-bitpacked-hnsw`:

1. **`RawU32Store`** — flattened `Vec<u32>` with a `u32` length prefix; the
   hnswlib/FAISS baseline.
2. **`DeltaVarintStore`** — sort each list, delta-encode, LEB128-varint.
3. **`BitPackedStore`** — sort, delta, fixed-width bit-pack with a per-list
   `bits` header.

The trait exposes O(1) random-access decode of any node's adjacency, so it is
a drop-in replacement for HNSW's neighbor-array reads. Insertion order is
not preserved (delta encoding requires sorted lists) — HNSW search does not
depend on order, but callers that do can keep a small side-buffer.

**Measured** (2026-07-07, Apple Silicon, release build, N/M configurations):

| Config      | raw bpe | varint bpe | bitpack bpe | bitpack ratio |
|-------------|---------|------------|-------------|---------------|
| 1 k × 16    | 4.50    | 1.49       | 1.45        | 3.11×         |
| 10 k × 16   | 4.50    | 2.19       | 1.86        | 2.42×         |
| 100 k × 16  | 4.50    | 2.41       | 2.28        | 1.97×         |
| 10 k × 32   | 4.25    | 1.85       | 1.56        | 2.72×         |
| 10 k × 64   | 4.13    | 1.53       | 1.36        | 3.03×         |

Decode latency 40–130 ns per node (bit-packed) — <5 % of a typical
384-dimensional distance computation. Full reproducer:
`cargo run --release -p ruvector-bitpacked-hnsw --bin bpk-bench`.

## Consequences

**Positive**
- ~3× graph-memory reduction on realistic HNSW configurations (M ≥ 16, N ≥ 10 k).
- Enables larger indexes to remain fully in-memory on the same hardware and
  reduces cold-start I/O for disk-resident indexes.
- Pure safe Rust, zero deps — trivial to audit and vendor into
  `ruvector-core`.
- Trait-based; existing search code changes only at the read-adjacency call
  site.

**Negative**
- Delta encoding forces sorted neighbor order; any caller that reads
  "position 0 is the most recently promoted neighbor" must be refactored.
- Build cost is ~15× the raw layout (13 ms for 100 k × 16). Streaming inserts
  need a hot delta layer + lazy compaction (LSM-style — see
  `crates/ruvector-lsm-ann/`).
- Per-list overhead (offset u32, length u16, bits u8 for bit-pack) dominates
  for very small graphs (< ~4 k nodes) — raw layout is competitive there.

**Neutral / follow-up**
- Snapshot format is versionless little-endian TLV in the PoC; needs a
  versioned header before promotion.
- SIMD unpack and PFOR-Delta are known 1.5–2× improvements left on the table.
- Graph reordering (WebGraph / ParlayANN) is orthogonal and multiplicative —
  another 1.2–1.4× on top.

## Alternatives considered

1. **Do nothing / rely on OS page cache.** — Cheapest, but leaves 60–70 % of
   the graph footprint as pure waste. Also ignored by disk-resident indexes
   where memory savings translate directly to reduced I/O.
2. **`u16` neighbor ids** — halves memory for N < 65 536 only; a dead end for
   any production-sized index.
3. **General-purpose compression (zstd/lz4)** on the flattened graph — great
   ratio but destroys random access; a single `decode` becomes O(block size),
   not O(deg). Unusable in the HNSW search loop.
4. **PFOR-Delta / SIMD-BP128 (Lemire 2015)** — better than the fixed-width
   bit-pack when a small number of outlier deltas dominate. Documented as a
   follow-up; adds implementation surface and requires either a portable-SIMD
   dep or hand-written intrinsics.
5. **Graph reordering only (ParlayANN)** — improves cache locality but does
   not reduce bytes on the wire. Compose with this ADR, don't substitute.

## Rollout

1. Land the PoC crate on the nightly research branch (this ADR).
2. If the numbers hold under real HNSW-built graphs (SIFT1M, GIST1M): promote
   to `crates/ruvector-graph-codec` with a versioned snapshot header, a
   `graph-codec` feature flag in `ruvector-core`, and wire into
   `ruvector-snapshot`.
3. Add SIMD unpack and PFOR-Delta as separate ADRs once the trait surface is
   stable.
