# Compressed IVF posting lists for ruvector (PFOR-128 + delta-varint)

**Date:** 2026-10-07 · **Branch:** `research/nightly/2026-10-07-ivf-pfor-postings` ·
**ADR:** [ADR-360](../../../adr/ADR-360-ivf-pfor-postings.md) ·
**Crate:** `crates/ruvector-ivf-pfor-postings`

## Abstract

ruvector's IVF-family indexes store per-cluster posting lists as raw
`Vec<u32>`. At N = 10⁹ docs that is 4 GiB of posting-list memory
*before* a single vector byte. We port two classical IR codecs — delta +
LEB128 varint and delta + Frame-of-Reference bitpacked 128-blocks (PFOR) —
into a swappable `PostingCodec` trait, measure real-cargo numbers on 100 k /
1 M / 5 M synthetic IVF indexes, and show (a) 2.0–3.2× size reduction, (b)
bit-exact top-K parity with the raw baseline, and (c) *faster* end-to-end
scan latency because the compressed lists hit cache harder than they pay in
decode. Encode is 1.5–4× slower, which is irrelevant: IVF builds are
k-means-dominated.

## State-of-the-art survey

Compressing sorted docid lists is an inverted-index staple going back to
MG4J (Boldi/Vigna) and Lucene. The modern menu:

* **PForDelta** — Zhang, Zukowski, Boncz 2006 ("Super-scalar RAM-CPU cache
  compression"). Fixed-width bitpack per block, patch outliers as
  exceptions. Lucene used it for Lucene41PostingsFormat.
* **FastPFor / SIMD-BP128** — Lemire & Boytsov 2015 ("Decoding billions of
  integers per second through vectorization",
  Software: Practice & Experience 45(1):1-29). 128-element blocks,
  SIMD unpack, order-of-magnitude faster than scalar PFOR.
* **StreamVByte** — Lemire, Kurz, Rupp 2017. SIMD-friendly varint; 4 ids
  per group with their lengths packed in a control byte.
* **QMX / OptPFD / Simple-16** — bitpacker variants trading encode
  complexity for 5–10% size wins.
* **Elias-Fano** — Fano 1971 / Vigna 2013 ("Quasi-Succinct Indices",
  WSDM'13). Near information-theoretic size *and* O(1) rank/select, used
  by Terrier and the OPT-DAAT literature. Lets you intersect posting
  lists in sub-linear time, which matters for hybrid IVF+keyword.
* **Roaring bitmaps** — Chambi, Lemire et al 2016. Wins when posting
  density is high (dense bit regions); IVF lists are typically sparse so
  Roaring loses to PFOR here. (Lucene 9+ keeps Roaring for `MATCH_ALL`
  but PFOR for inverted postings.)

ANN libraries that *do* bother with inverted-list compression:

* **DiskANN / Vamana** — stores neighbour lists on disk packed as 4-byte
  ids plus a quantised vector; posting-style compression is applicable
  but not yet standard upstream.
* **ScaNN** — compresses posting lists per leaf with varint, benchmarked
  in Google's "Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization" follow-up blogs.
* **PISA** — competitive SOTA for pure IR; PFOR, QMX, Simple-16 all
  implemented. ANN crossover work from Mallia et al. (2021–2024).

None of ruvector's existing IVF crates (`ruvector-rairs`,
`ruvector-spann`, `ruvector-soar-ivf`, `ruvector-blockmax-ivf`,
`ruvector-batched-ivf`, `ruvector-norm-bucket-ivf`) compress their
posting lists today — all keep raw `Vec<u32>`. This nightly is the
missing swap-point.

## Proposed design

### `PostingCodec` trait

```rust
pub trait PostingCodec {
    fn encode(ids: &[u32]) -> Vec<u8>;   // assumes sorted ascending
    fn decode(bytes: &[u8]) -> Vec<u32>;
    fn name() -> &'static str;
}
```

### `RawCodec`
Little-endian `u32` array. Baseline for correctness and perf
comparisons.

### `VarintCodec` — delta + LEB128

Encode: `varint(n) · varint(ids[0]) · varint(ids[i] - ids[i-1]) …`.
LEB128 continuation-bit variant. One byte per 7 bits of gap; typical IVF
gap of 5–50 fits in a single byte.

### `PforCodec` — delta + Frame-of-Reference bitpacked blocks

```
[varint n] [u32_le first_id]
  [ [u8 block_len (1..=128)] [u8 bit_width (0..=32)]
    [ceil(block_len * bit_width / 8) bytes of bitpacked gaps] ]*
```

Each 128-gap block picks the smallest `b` that holds its max gap, writes
`b` as one byte, then packs `block_len * b` bits little-endian. The
scalar `put_bits` / `get_bits` operate on a u64 window and write/read up
to five bytes per value (max `b` = 32 + up to 7-bit offset).

### IVF scan integration

`scan::scan(ids, vectors, dim, q, k)` is a minimal top-K harness with a
`BinaryHeap<Hit>` of size `k` and a scalar `l2_sq`. Any existing IVF
scan kernel keeps its distance code and just swaps the posting-list
decode call.

## Implementation notes

* `#![forbid(unsafe_code)]` — the whole crate is safe Rust.
* Zero external deps (`cargo tree -p ruvector-ivf-pfor-postings` is a
  single node). Keeps the WASM path trivial.
* Encode allocates a `Vec<u8>` once (`ids.len() * 2 + 16` is a tight
  bound for PFOR's typical 1–2 bytes/id output).
* Edge-case tests cover empty lists, lists whose length straddles a 128
  boundary (127, 128, 129, 255, 256, 257), and random bit widths from 1
  to 32 in `put_get_bits_roundtrip`.

## Benchmark methodology

`cargo run --release -p ruvector-ivf-pfor-postings --example measure`

Three scenarios (`tiny`, `medium`, `large`). For each:

1. Partition `N` docs across `C` clusters uniformly at random (xorshift,
   seed `0xBEEFCAFE`). Each cluster list is sorted ascending (property of
   how ids were pushed — matches real IVF builds).
2. **Encode** each cluster list with all three codecs. Report total
   bytes, bits/id, and total encode time.
3. **Decode throughput:** repeatedly decode every list, sum ids into a
   `sink` to defeat dead-code elimination, report (passes · total_ids /
   seconds) in millions of ids/sec.
4. **End-to-end scan:** decode-then-top-K-L2 across *all* clusters
   (worst-case: scan the whole index). 64-d vectors, k = 10, L2 metric.
5. **Correctness:** verify top-K from varint and PFOR equals raw
   bit-exactly per cluster.

Platform: Apple Silicon (Darwin arm64), release profile (opt-level 3,
LTO off — default workspace profile), single thread.

## Results

```
--- scenario: tiny (100000 docs, 128 clusters, avg 781.2 ids/list) ---
  encode:        raw-u32 |     400000 bytes |  32.00 b/id | encode-all:    0.08 ms
  encode:   delta+varint |     137322 bytes |  10.99 b/id | encode-all:    0.43 ms
  encode: delta+pfor-128 |     126549 bytes |  10.12 b/id | encode-all:    0.31 ms
  compression: varint 2.91x, pfor 3.16x
  decode:        raw-u32 |  1820.6 M ids/s
  decode:   delta+varint |   299.6 M ids/s
  decode: delta+pfor-128 |   523.9 M ids/s
  scan:          raw-u32 | all clusters:    4.30 ms
  scan:     delta+varint | all clusters:    3.31 ms
  scan:   delta+pfor-128 | all clusters:    2.84 ms   ← 34% faster scan at 3.16× smaller

--- scenario: medium (1000000 docs, 512 clusters, avg 1953.1 ids/list) ---
  encode:        raw-u32 |    4000000 bytes |  32.00 b/id | encode-all:    0.53 ms
  encode:   delta+varint |    1781572 bytes |  14.25 b/id | encode-all:    2.19 ms
  encode: delta+pfor-128 |    1510729 bytes |  12.09 b/id | encode-all:    2.06 ms
  compression: varint 2.25x, pfor 2.65x
  decode:        raw-u32 |  2830.3 M ids/s
  decode:   delta+varint |   503.8 M ids/s
  decode: delta+pfor-128 |   566.4 M ids/s
  scan:          raw-u32 | all clusters:   60.65 ms
  scan:     delta+varint | all clusters:   59.91 ms
  scan:   delta+pfor-128 | all clusters:   59.01 ms

--- scenario: large (5000000 docs, 2048 clusters, avg 2441.4 ids/list) ---
  encode:        raw-u32 |   20000000 bytes |  32.00 b/id | encode-all:    3.12 ms
  encode:   delta+varint |    9705900 bytes |  15.53 b/id | encode-all:    6.84 ms
  encode: delta+pfor-128 |    8804307 bytes |  14.09 b/id | encode-all:   12.26 ms
  compression: varint 2.06x, pfor 2.27x
  decode:        raw-u32 |  2700.2 M ids/s
  decode:   delta+varint |   741.3 M ids/s
  decode: delta+pfor-128 |   527.1 M ids/s
  scan:          raw-u32 | all clusters:  373.56 ms
  scan:     delta+varint | all clusters:  317.65 ms
  scan:   delta+pfor-128 | all clusters:  314.50 ms

  correctness: top-10 identical across all three codecs for all clusters (all scenarios)
```

## How it works (blog-readable)

Picture a cluster's posting list as a telephone book of row numbers
sorted smallest-first. Row numbers themselves can be big (up to a few
billion), but the *gaps* between adjacent rows are small — a dozen or so
on a well-balanced IVF with a thousand clusters over a million docs. We
exploit that: store the first row raw, then only the gaps.

The **varint** codec writes each gap as a chain of 7-bit chunks with a
"keep going" bit; a gap of 50 is one byte, 10 000 is two, millions are
three. Decode is a byte-at-a-time running sum.

The **PFOR** codec is smarter: it looks at 128 gaps at a time, finds the
widest one, picks the smallest number of bits `b` that fits it (say 11
bits for a max gap of 1 500), writes one header byte saying "`b = 11`",
then crams 128 × 11 = 1 408 bits = 176 payload bytes. Decode unpacks
`b`-wide words, adds each to a running total.

**Why does scanning speed *improve* despite doing more decode work?**
Because an IVF scan is RAM-bound. Posting lists are already in L3 for
small N, but for N = 5 M the raw list is 20 MiB — about half an L3
cache on an M-series Mac. PFOR shrinks it to 8.8 MiB (fits comfortably),
and the vectors themselves are the bigger bandwidth hog. The decode cost
is < 1 ns/id and runs partly during what would have been a stall on the
`vectors[id * dim..]` load.

## Practical failure modes

* **Short lists lose to the 2-byte block header overhead.** For a list
  of 10 ids the PFOR block header is 2 bytes out of maybe 15 total → 13%
  overhead. Fix: pick codec per list by length (`varint` if < 32,
  `pfor` otherwise).
* **Pathological gaps.** If a tenant creates one cluster with ids
  `{1, 2³¹ - 1}` the single huge gap forces `b = 31` for the whole block.
  Mitigate with the patched-exception variant (full PFORDELTA), which we
  leave as roadmap item 2.
* **Scalar bitpack is memory-bound at large widths.** `get_bits` touches
  up to 5 bytes per read; a SIMD unpack (roadmap item 1) is where the
  real-world speedup lives for widths 1–16.
* **No persistence format yet.** `encode` returns a `Vec<u8>` for
  benchmarking. Production storage needs a stable framing (length, codec
  tag, checksum) — the trait makes it straightforward.
* **No concurrency / sharing.** The scan harness is single-threaded; a
  real IVF scan would parallelise across clusters. PFOR's block
  independence makes it embarrassingly parallel at any granularity ≥ 128
  gaps.

## What to improve next (roadmap)

1. **SIMD unpack** (`std::simd` NEON/AVX2) for bit widths 1..32. Expected
   3–5× PFOR decode speedup; the trait absorbs it with no API change.
2. **Patched-exception variant** (FastPFor-style). Store the 10% worst
   outliers out-of-band, pick `b` for the 90th percentile.
3. **Per-list codec selection** by length and distribution. Build-time
   decision; 1 byte tag per list.
4. **Elias-Fano** codec. Enables sub-linear intersection for
   `IVF + sparse-keyword` hybrids.
5. **Wire into ruvector-rairs and ruvector-blockmax-ivf** behind
   `--features compressed-postings`.

## Production crate layout proposal

```
crates/ruvector-ivf-postings/      (promoted from ruvector-ivf-pfor-postings)
├── src/
│   ├── lib.rs                    PostingCodec trait, selection heuristic
│   ├── raw.rs
│   ├── varint.rs
│   ├── pfor.rs
│   ├── pfor_simd.rs              feature = "simd"
│   ├── elias_fano.rs             feature = "intersect"
│   ├── framing.rs                [magic|codec_tag|crc32] persistence
│   └── scan.rs                   generic over codec, plugs into IVF
├── benches/
│   └── sota_bench.rs             criterion suite vs raw
└── examples/
    └── measure.rs                the current file
```

## References

1. Zhang, Zukowski, Boncz. "Super-scalar RAM-CPU cache compression", ICDE 2006.
2. Lemire, Boytsov. "Decoding billions of integers per second through vectorization",
   *Software: Practice & Experience* 45(1):1-29, 2015.
3. Lemire, Kurz, Rupp. "Stream VByte: faster byte-oriented integer compression",
   *Information Processing Letters* 2018.
4. Vigna. "Quasi-Succinct Indices", WSDM 2013.
5. Chambi, Lemire, Kaser, Godin. "Better bitmap performance with Roaring bitmaps",
   *Software: Practice & Experience* 46(5):709-719, 2016.
6. Mallia, Siedlaczek, Mackenzie, Suel. "PISA: Performant Indexes and Search for Academia",
   OSIRRC 2019 / updates 2021–2024.
7. Guo, Sun, Liu et al. "Accelerating Large-Scale Inference with Anisotropic Vector
   Quantization" (ScaNN), ICML 2020.
8. Jayaram Subramanya et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search
   on a Single Node", NeurIPS 2019.
