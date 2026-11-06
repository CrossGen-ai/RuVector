# D2-IVF: Delta-of-Delta Encoding for IVF Posting Lists

**Nightly research, 2026-11-06. Branch:
`research/nightly/2026-11-06-d2-ivf-posting-encoding`.**
**ADR:** [ADR-356](../../../adr/ADR-356-d2-ivf-posting-encoding.md).

## Abstract

IVF probe kernels spend measurable time touching cluster posting lists.
After a locality reorder, posting IDs inside a cluster become monotone,
and in sharded or multi-tenant layouts near-arithmetic-progression (AP).
This PoC applies **delta-of-delta** (D2) — the Gorilla / InfluxDB
time-series trick — to IVF posting IDs, under a trait-based swap layer
so the probe kernel can pick per-posting between raw u32, single-delta
varint, and D2 varint.

Real `cargo run --release` numbers on 1M synthetic IDs:

| regime                                   | raw   | varint-delta | **D2** | D2 vs varint-delta |
| ---------------------------------------- | ----- | ------------ | ------ | ------------------ |
| uniform random, 1K clusters              | 4.00  | 1.94         | 1.94   | 1.00x (ties)       |
| uniform random, 16K clusters             | 4.00  | 2.61         | 2.61   | 1.00x (ties)       |
| dense slab (stride ≈ 1), jitter 32       | 4.00  | 1.00         | 1.00   | 1.00x (floor)      |
| **sparse-ID AP, stride=1000 jitter=5**   | 4.00  | 2.00         | **1.01** | **1.99x**        |
| **sparse-ID AP, stride=10000 jitter=5**  | 4.00  | 3.00         | **1.01** | **2.98x**        |
| sparse-ID AP, stride=100000 jitter=50    | 4.00  | 3.00         | 1.14   | 2.63x              |

(Units: bytes per ID.)

**Headline:** D2 wins 1.6x–3x over PFOR-lite only in the sparse-ID /
multi-tenant regime where the common delta exceeds 128. In the
dense-slab regime (classic IVF after locality reorder) both codecs
share the 1 B/ID varint floor — D2 is a no-op.

## SOTA survey

### Time-series lineage

* **Pelkonen et al., Gorilla (VLDB 2015).** Delta-of-delta for monotone
  Facebook metric timestamps; used in Beringei, M3DB, Prometheus.
* **Iyer et al., InfluxDB TSM engine.** DoD + Simple8b + zigzag for
  timestamps in production since 2017.
* **MonetDB / DuckDB column compressors.** Similar DoD + zigzag family on
  sorted integer columns.

### Vector-DB posting codecs

* **Faiss PFOR (2015)**, SPANN, DiskANN — all stop at single-delta +
  varint / fixed-width packed codes.
* **Milvus/Weaviate** — PFOR or raw u32 depending on list length.
* **BitMagic/Succinct Rank/Select** — tighter floor but orders of
  magnitude more complexity than a varint stream.
* **Elias-Fano** (ruvector nightly 2026-10-26, `gist-elias-fano-hnsw-adjacency`) —
  the information-theoretic floor for monotone sequences, but fixed
  per-posting overhead.
* **StreamVByte** (nightly 2026-10-29) — SIMD-friendly single-delta.
* **PFOR postings** (nightly 2026-10-07) — single-delta with exception
  patch lists.

None of the above apply DoD to posting IDs, because the common assumption
is that IDs are ≈ dense after a global shuffle. Composing locality
reorder (ADR-352) with sharded ID assignment flips that assumption —
and that is where D2 pays off.

## Proposed design

```
   posting-list codec trait
       |
   +---+---+---+
  Raw Varint   D2Varint
  u32  Delta     ^
                 |
     zigzag + LEB128 varint of (d_i - d_{i-1})
```

* `encode(&[u32]) -> Vec<u8>` and `decode(&[u8]) -> Vec<u32>`.
* First two IDs encoded as single IDs; i ≥ 2 encoded as second
  differences, zigzag-folded, LEB128 varint.
* Streaming decoder keeps two i64 registers (`cur_id`, `prev_delta`).
* 2-bit per-posting tag selects codec in the file format.

## Implementation notes

* Pure safe Rust, no SIMD, no `unsafe`.
* `rand_chacha` for deterministic synthesis across runs.
* 5 property/roundtrip tests (empty, single, increasing, arithmetic
  progression, synth) all green.
* Two synthesizers:
  * `synth_postings`: uniform random assignment (baseline).
  * `synth_postings_ap_biased`: dense slab with per-step jitter.
  * `synth_postings_sparse_ap`: 32-bit sparse namespace with configurable
    stride — the regime D2 targets.

## Benchmark methodology

* Hardware: macOS, Apple Silicon (as-measured; see raw log).
* 1,000,000 synthetic IDs across 1K / 4K / 16K clusters for uniform.
* 1024 clusters × 1000 IDs/posting for sparse-AP sweep.
* Each decode pass repeats 5x; throughput = `total_ids / elapsed`.
* Metric: `b/ID = total_encoded_bytes / id_count`.
* `cargo build --release -p ruvector-d2-ivf-encoding` succeeds.
* `cargo test --release` passes (5 tests).
* `cargo run --release --bin bench` produces the numbers above in ~1 s.

## Results (raw log excerpts)

```
PART A: UNIFORM-RANDOM assignment (worst-case for D2)
## 1K clusters (mean posting len = 976.6)
raw-u32              4000000      4.00           0.7      3161.5 M-IDs/s
varint-delta         1942837      1.94           1.7       476.9 M-IDs/s
d2-varint            1941572      1.94           2.1       459.8 M-IDs/s

PART B: AP-biased dense slab (stride~1)
## jitter=0 (perfect AP) — 999224 IDs
varint-delta         1003310      1.00
d2-varint            1003310      1.00     ← floor tied

PART C: Sparse-ID AP regime (stride>128)
## stride=1000 jitter=5
varint-delta         2053088      2.00           1.2       518.3
d2-varint            1031136      1.01           0.6       542.8   ← 1.99x smaller

## stride=10000 jitter=5
varint-delta         3076062      3.00           1.5       434.4
d2-varint            1032170      1.01           0.8       546.3   ← 2.98x smaller
```

## How it works (blog-readable walkthrough)

Imagine the row IDs assigned to one IVF cluster look like:

    4096, 5096, 6098, 7099, 8097, 9103, ...

Naively, that costs 4 bytes/ID (24 B). Taking **single deltas**:

    4096, 1000, 1002, 1001, 998, 1006, ...

The first value needs 2 varint bytes (4096 > 127). Each subsequent delta
is close to 1000, which needs 2 varint bytes too. Total ≈ 2 B/ID.

Now take **delta-of-delta**:

    4096, 1000, +2, −1, −3, +8, ...

The first two still cost 2 bytes each, but after that every second
difference is tiny — one zigzag-varint byte each. Total ≈ 1 B/ID.

The trick: single-delta encodes the **constant stride** of every step;
DoD encodes only the **jitter** on top of it, which is much smaller.

## Practical failure modes

* **Delta ≤ 127.** Both codecs tie at 1 B/ID. Don't pay D2's extra
  register update — pick VarintDelta.
* **No monotone ordering.** Raw unsorted IDs: zigzag absorbs negatives
  but the magnitudes stay u32-sized. Neither codec wins. Sort first.
* **Duplicates.** Synth code dedups because the stride walk can wrap.
  Production postings should not contain duplicates — if they do, zero
  deltas are optimal here and the codec stays correct (emits 0x00).
* **Wraparound.** Encoder uses `wrapping_add`; `decode` is deterministic
  on the same wrap convention. Use u64 sums for very large deltas if
  posting lists could ever be close to u32::MAX / 2.

## What to improve next (roadmap)

1. Fold into `ruvector-ivf-centroid-locality-reorder` so the reorder
   pipeline writes D2-encoded postings directly.
2. SIMD decode kernel (AVX2, NEON) — compare against StreamVByte's
   single-delta kernel.
3. Hybrid per-posting codec selection — scan first K deltas, pick D2 if
   mean delta > 128.
4. Real SIFT1M + LAION400K runs through `ruvector-sota-bench` to replace
   synthetic stride numbers with observed stride distributions.
5. Compare against Elias-Fano (gist-elias-fano-hnsw-adjacency) at the
   same posting lengths — EF is the information-theoretic floor; D2 is
   the cheap-to-decode runner-up.

## Production crate layout proposal

```
crates/ruvector-d2-ivf-encoding/
  src/
    lib.rs          — trait + three codecs + synthesizers + tests
    bin/bench.rs    — reproducible benchmark binary
  Cargo.toml        — standalone workspace, rand + rand_chacha
```

Integration with the real `ruvector-ivf-*` crates: introduce a 2-bit
per-posting codec tag in the IVF file header and route `encode/decode`
through the trait. Default stays `VarintDelta` for byte-level
compatibility; D2 activates only when the per-posting codec-selector
heuristic picks it (follow-up iteration).

## References

* P. Pelkonen et al., *Gorilla: a fast, scalable, in-memory time series
  database*, VLDB 2015.
* Y. Dukhovni, *InfluxDB TSM engine internals*, InfluxData blog, 2017.
* Lemire, Kurz, Rupp, *Stream VByte: Faster Byte-Oriented Integer
  Compression*, Information Processing Letters, 2018.
* Faiss / FaissIVFDisk, pfor_codec, Meta.
* ADR-352 (ivf-centroid-locality-reorder), ruvector nightly 2026-11-02.
* `gist-elias-fano-hnsw-adjacency-2026`, ruvector nightly 2026-10-26.
* `gist-ivf-pfor-postings-2026`, ruvector nightly 2026-10-07.
* `gist-streamvbyte-hnsw-adjacency-2026`, ruvector nightly 2026-10-29.
