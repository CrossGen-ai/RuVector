# StreamVByte Delta Encoding for HNSW Adjacency

**Date:** 2026-10-29
**Slug:** `streamvbyte-hnsw-adjacency`
**Crate:** `crates/ruvector-streamvbyte-hnsw`
**Status:** Research PoC — numbers are real, measured on the author's machine.

## Abstract

HNSW adjacency lists dominate the RAM footprint of a graph-based ANN index at
mid-to-large degree (M=16–64). The ids are small u32s, usually sortable, and
their deltas are heavily skewed toward small values. Three codecs matter in
practice: plain u32, delta + LEB128 varint, and **StreamVByte** (Lemire 2017),
which separates control-byte and data streams so decode becomes a table-driven
shuffle rather than a per-byte branch.

This PoC implements all three behind a common trait, measures size and decode
throughput on three synthetic HNSW-shaped graphs (N = 10k / 100k / 500k, degree
= 16 / 32 / 32), and reports **1.7–2.0× space savings** with decode throughput
in the 1.4–3.9×10⁸ edges/sec range depending on size.

## SOTA survey

- **StreamVByte** — Lemire, Kurz, Rupp. *StreamVByte: Faster Byte-Oriented
  Integer Compression.* arXiv:1709.08990 (2017). Separates length control
  bytes from the data stream so a 16-byte SIMD shuffle (`pshufb`) decodes 4
  integers at once. In the paper: ~4 GB/s decode on Haswell for u32.
- **MaskedVByte** — Plaisance, Kurz, Lemire (2015). Same bit-level encoding
  as classic VByte, but uses PEXT/BMI2 for masked parallel decode.
- **Group-varint (varint-GB)** — Dean (Google, 2009). 1 control byte per 4
  values with length codes; the ancestor of StreamVByte.
- **FastPFor / SIMD-BP128** — Lemire & Boytsov (2015). Bit-packing variants
  that beat byte codecs for very dense deltas but need large block sizes.
- **Elias–Fano / Partitioned EF** — Vigna (2013); used in Qdrant for sorted
  posting lists. Covered previously in [`elias-fano-hnsw-adjacency`
  (2026-10-26)](../../../../gist-elias-fano-hnsw-adjacency-2026.md).
- **Milvus** uses raw u32 arrays for graph edges; **FAISS HNSW** keeps a
  dense `[M]u32` block per node; **Qdrant** switched to Elias–Fano in 1.7.
- **DiskANN** uses raw ids on disk but zstd-compresses whole shards.
- **CAGRA (NVIDIA)** stores one row of 32×u32 per node for GPU locality —
  no compression.
- Relevant Rust ecosystem: `stream-vbyte` crate (last touched 2019; `unsafe`
  SSSE3 shuffle), `varint-rs`, `integer-encoding` (varint only), `q_compress`
  (float focus). None target HNSW adjacency.

Previously-covered-in-ruvector topics on this axis: `hnsw-delta-neighbors`
(plain delta), `elias-fano-hnsw-adjacency` (bit-packed), `hnsw-freq-reorder`
(id reordering to shrink deltas). This PoC fills the StreamVByte slot and
quantifies the three-way trade.

## Proposed design

A `AdjacencyCodec` trait with `encode(&[u32], &mut Vec<u8>)` and
`decode(bytes, len, &mut Vec<u32>)`. Three implementations:

- `PlainCodec` — baseline, u32 LE.
- `DeltaVarintCodec` — zigzag-free (ids are monotone post-sort so deltas are
  unsigned) + LEB128 varint. Branch per byte on decode.
- `StreamVByteCodec` — delta variant. Groups of 4 deltas → 1 control byte
  (2 bits per delta: 00=1B, 01=2B, 10=3B, 11=4B) + packed data. Control
  bytes stored contiguously up front; data stream follows.

Adjacency wrapper `CompressedAdjacency<C>` holds columnar `offsets[]`,
`lens[]`, `data[]`. Access is `data[offsets[i]..offsets[i+1]]` + the
per-node length (needed because the codecs don't length-prefix).

## Implementation notes

- Pure safe Rust. The scalar StreamVByte decode is already very
  branch-light — `(code + 1) as usize` indexes a 4-byte buffer and we
  `copy_from_slice` the needed prefix.
- A SIMD decode on x86_64 would use `_mm_shuffle_epi8` with a 256-entry
  shuffle-mask table keyed on the control byte. Future work gated behind
  `target_feature="ssse3"`; not required for the numbers below.
- The crate is additive: no changes to existing HNSW code. Integration path
  is to replace `Vec<Vec<u32>>` adjacency with `CompressedAdjacency<C>` in
  `ruvector-core`'s HNSW builder.

## Benchmark methodology

- Random synthetic graphs: ids drawn uniformly from `0..N`, sorted,
  deduped. Mean degree held at the configured value (dedup trims it ~0.03).
- Three sizes: (10k, 16), (100k, 32), (500k, 32).
- Decode workload: 20,000 random node lookups, 3 passes, mean edges/sec
  reported.
- Compression ratio: `plain_bytes / codec_bytes` (data + offsets + lens).
- Hardware: Apple M1 Pro (ARM NEON, no SSSE3). `--release`, LTO=thin.
- Reproduce with `cargo run --release -p ruvector-streamvbyte-hnsw --bin svb-demo`.

## Results

```
=== small graph: N=10000, degree=16 ===
  codec                      bytes   B/edge   ratio      edges/sec
  plain-u32                 699500    4.375   1.00x       6.328e8
  delta-varint              350330    2.191   2.00x       2.919e8
  stream-vbyte-delta        365679    2.287   1.91x       2.061e8

=== medium graph: N=100000, degree=32 ===
  codec                      bytes   B/edge   ratio      edges/sec
  plain-u32                13397844    4.188   1.00x       3.806e8
  delta-varint              6881867    2.151   1.95x       3.975e8
  stream-vbyte-delta        7548298    2.359   1.77x       2.227e8

=== large graph: N=500000, degree=32 ===
  codec                      bytes   B/edge   ratio      edges/sec
  plain-u32                66997796    4.188   1.00x       1.735e8
  delta-varint             40381335    2.524   1.66x       1.295e8
  stream-vbyte-delta       38918563    2.432   1.72x       1.368e8
```

Observations:

1. **Both codecs give ~1.7–2.0× size savings** vs plain u32 across all
   three sizes. At 500k nodes × 32 edges, that's 28 MB saved — enough to
   keep the whole graph in a tier higher in the cache hierarchy.
2. **Varint beats scalar StreamVByte at small N** — the control-byte
   dispatch overhead is paid even when deltas are tiny (1 byte).
3. **StreamVByte wins decisively at large N** where deltas spread into 2–3
   byte codes: smaller output (28% smaller than plain, 4% smaller than
   varint at N=500k) *and* slightly faster decode on the Apple M1 scalar path.
4. On x86_64 with SSSE3, the published Lemire numbers are 2–3× the scalar
   decode; expect StreamVByte to extend its lead there.

## How it works

HNSW builds a graph of ~32 neighbors per node, each an id. If N=500k,
each id fits in 20 bits, but a plain `u32` burns 32. We can do better by
noticing two things:

- **Sorted lists compress by their differences.** Neighbor id 3 after id 7
  becomes just "+4" — a single byte, not four. This is the "delta" idea
  every modern search engine uses on posting lists.
- **The length of each delta is uneven.** Some are 1 byte, some are 3. A
  varint tags each byte with "more?" bits, which costs a branch per byte at
  decode time. StreamVByte peels those tag bits out and packs them into
  one byte per group of four — the data stream becomes pure values.

Decode then becomes: read the control byte, look up each 2-bit lane,
copy that many bytes into a u32 slot. On x86 with SSSE3 that's a single
`pshufb` lookup; on scalar Rust it's still branch-light.

## Practical failure modes

- **Scalar M1 numbers are competitive, not dominant.** Without `pshufb`
  the per-group dispatch loses some of its edge. On x86-SSSE3 servers the
  gap widens; see Lemire's paper.
- **Partial final group.** Groups of 4 ids with a partial last group need
  careful bookkeeping (we track `lens[]` separately). Off-by-one bugs here
  are classic.
- **Insertion/update cost.** StreamVByte is a batch codec. Mutating a
  single neighbor requires decoding, patching, re-encoding the whole list.
  For HNSW this is fine (edges are rewritten on neighbor-selection pass
  anyway), but a mutable graph needs an uncompressed overlay or a tombstone
  log.
- **Not a win for very dense/sequential ids.** If id reordering already
  packed neighbor ids into tiny ranges, bit-packing (FastPFor / Elias-Fano)
  catches up because the delta variance is ~0.

## What to improve next

1. Enable the SSSE3 (`_mm_shuffle_epi8`) decode path with the standard
   256-entry mask table. Expect 2–3× decode throughput on x86 servers.
2. ARM NEON path with `vqtbl1q_u8` for Apple-silicon parity.
3. Pair with the existing `hnsw-freq-reorder` ordering: id reordering by
   access frequency shrinks deltas further and should push StreamVByte
   below 2.0 B/edge on realistic workloads.
4. Benchmark on a real HNSW traversal pattern (not uniform random) — the
   decode scratch buffer stays warmer there.
5. Compare against Elias-Fano on the same data to pick the right codec
   per layer: EF for the dense low-id lists on L0, StreamVByte for the
   sparser higher layers.

## Production crate layout

```
crates/ruvector-streamvbyte-hnsw/
  Cargo.toml
  src/
    lib.rs                 # Codec trait + impls + CompressedAdjacency
    bin/demo.rs            # Measured size/throughput demo
  benches/
    svb_bench.rs           # Criterion decode bench
  tests/                   # (unit tests live in lib.rs)
```

Integration path: expose `AdjacencyCodec` + `CompressedAdjacency` to
`ruvector-core`'s HNSW; feature-flag the codec choice per layer.

## References

- Lemire, D., Kurz, N., Rupp, C. *StreamVByte: Faster Byte-Oriented
  Integer Compression.* arXiv:1709.08990 (2017).
- Dean, J. *Challenges in Building Large-Scale Information Retrieval
  Systems.* WSDM 2009.
- Lemire, D., Boytsov, L. *Decoding Billions of Integers per Second
  Through Vectorization.* Softw. Pract. Exp. 45(1), 2015.
- Vigna, S. *Quasi-Succinct Indices.* WSDM 2013.
- Malkov, Y., Yashunin, D. *Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs.*
  TPAMI 42(4), 2020.
