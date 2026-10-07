# ADR-360: Delta + Frame-of-Reference Compression for IVF Posting Lists

## Status

**Proposed.** 2026-10-07. PoC in `crates/ruvector-ivf-pfor-postings`.
Numbers in this ADR come from `cargo run --release -p ruvector-ivf-pfor-postings
--example measure` on Darwin arm64 (Apple Silicon, release profile) and are
reproducible from the branch `research/nightly/2026-10-07-ivf-pfor-postings`.

## Context

ruvector has many IVF-shaped indexes (`ruvector-rairs`, `ruvector-spann`,
`ruvector-soar-ivf`, `ruvector-norm-bucket-ivf`, the batched-IVF and
blockmax-IVF nightlies). Every one of them stores, per cluster, a *posting
list* of `u32` document ids. At N = 10⁹ docs that is **4 GiB of posting-list
bytes alone**, before any vector payload. Because the ids are produced in
insertion order they are already sorted inside each list, which is the exact
shape classical inverted-index compressors assume.

Information-retrieval systems (Lucene, Terrier, PISA) have been compressing
sorted docid lists for twenty years with delta + bit-packed codecs — PFOR,
PForDelta, Simple-16, StreamVByte, QMX. None of ruvector's IVF variants
currently use them: posting lists are stored as raw `Vec<u32>`.

## Decision

Introduce a `PostingCodec` trait with three initial implementations:

1. **`RawCodec`** — baseline, little-endian `u32` array.
2. **`VarintCodec`** — delta + unsigned LEB128 varint; byte-aligned,
   dependency-free, portable.
3. **`PforCodec`** — delta + 128-element fixed-width bitpacked blocks
   (one bit-width byte per block, payload sized to `ceil(block_len * b / 8)`
   bytes). The first id of a list is stored raw to anchor the running sum.

The crate is a leaf dependency (no inner-crate deps) and keeps every file
well under 500 lines. All three codecs implement `encode` and `decode`
returning a `Vec<u32>`, so any existing IVF scan kernel can drop the
decoded ids straight in.

### Measured results (Apple Silicon, release)

| Scenario (N docs, C clusters) | raw bytes | varint b/id | pfor b/id | varint × | pfor × | raw decode M/s | varint decode M/s | pfor decode M/s | raw scan ms | varint scan ms | pfor scan ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 k, 128 clusters | 400 k | 10.99 | **10.12** | 2.91× | **3.16×** | 1820 | 300 | 524 | 4.30 | 3.31 | **2.84** |
| 1 M, 512 clusters   | 4.0 MB | 14.25 | **12.09** | 2.25× | **2.65×** | 2830 | 503 | 566 | 60.65 | 59.91 | **59.01** |
| 5 M, 2048 clusters  | 20 MB | 15.53 | **14.09** | 2.06× | **2.27×** | 2700 | 741 | 527 | 373.56 | 317.65 | **314.50** |

Top-10 identity against the raw baseline is verified bit-exact for every
cluster in every scenario (see the correctness line in `examples/measure.rs`).

### Why `pfor` is slower-to-decode but faster-to-scan at scale

Varint has a trivially predictable byte-level inner loop; PFOR pays a
bit-unpack cost per element. In isolation, decode throughput for varint
beats PFOR at `large`. But end-to-end (decode + 64-dim L2 top-K) PFOR still
wins because:

* Compressed footprint is 1.17× smaller than varint and 2.27× smaller than
  raw, so more of the posting list stays resident in L2/L3 while we are
  also chasing cold vector rows.
* The scan is vector-dominated (L2 kernel per doc), so decode latency is
  partially hidden; what matters is how much RAM bandwidth posting I/O
  takes.

## Consequences

**Positive.**
- Posting-list memory drops ≈2–3× for free; at N = 10⁹ docs that is ≈2.3 GiB
  back on PFOR vs raw, with no recall change (bit-exact top-K).
- End-to-end scan is *faster*, not slower, at every tested N — compression
  pays its own decode budget.
- `PostingCodec` is a stable swap point: later codecs (StreamVByte, SIMD-BP128,
  QMX, Elias-Fano for `rank`/`select`) plug in without touching IVF scans.

**Negative / cost.**
- Encode is 1.5–4× slower than the raw baseline. IVF builds already dominate
  by k-means and quantizer training, so this is noise, but note it.
- PFOR's current scalar `put_bits` / `get_bits` are not SIMD; a vectorised
  128-wide unpack (Lemire's `simdcomp`) is left as future work (see roadmap).
- The 1-byte-per-block overhead means PFOR regresses to raw-ish bits/id if
  the average list is only a handful of ids. For short lists varint wins and
  the trait makes picking per-list a one-liner.

## Alternatives

- **StreamVByte** (Lemire, 2017). Four 2-bit length tags packed into one
  byte per 4-id group, then the data bytes. Fastest known pure-SIMD varint;
  likely faster than our LEB128 by 2–3×. Deferred: adds SIMD-dispatch
  machinery; this PoC stays scalar.
- **SIMD-BP128** / **FastPFor**. Fixed 128-element bit-packed blocks with
  SIMD unpack and exception patching. Smaller *and* faster than our PFOR.
  Deferred for the same SIMD reason; the trait is ready for it.
- **Elias-Fano**. Monotonically-encoded sorted lists supporting O(1)
  `rank`/`select` with near-information-theoretic size. Useful when IVF
  wants cross-list intersections (hybrid retrieval). Deferred as a separate
  nightly.
- **Roaring bitmaps**. Great for *dense* posting lists; IVF lists are
  typically 10³–10⁵ ids out of 10⁷+ docs, so sparsity wins and roaring
  tends to lose to PFOR on this shape.

## Roadmap (if adopted)

1. SIMD unpack for 1/2/3/…/32-bit fixed-width blocks (`std::simd` or
   `wide`) — expected 3–5× PFOR decode speedup.
2. Patched-exception variant (full PFORDELTA): store the 10% worst
   outliers as exceptions, drop the common bit-width.
3. Codec selection per list based on list length (`varint` ≤ 128, else
   `pfor`), driven by a cheap length heuristic at build time.
4. Wire `PostingCodec` into `ruvector-rairs` and `ruvector-blockmax-ivf`
   behind a `--features compressed-postings` flag; keep raw as the
   default for one release.
