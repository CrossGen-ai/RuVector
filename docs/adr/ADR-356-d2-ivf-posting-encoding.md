# ADR-356: Delta-of-Delta encoding for IVF posting-list IDs (D2-IVF)

**Status:** Proposed (nightly research PoC)
**Date:** 2026-11-06
**Related:** ADR-352 (ivf-centroid-locality-reorder), prior nightlies on
`ivf-pfor-postings`, `streamvbyte-hnsw-adjacency`, `elias-fano-hnsw-adjacency`.

## Context

IVF posting lists hold the row IDs assigned to each coarse cluster. Today
ruvector uses raw `Vec<u32>` (4 B/ID) or PFOR-style delta encoding (~2 B/ID
after reorder) for these lists. Memory and cache footprint of postings are
the dominant cost at probe time for large-K recall workloads, not the PQ
ADC itself — on `ivf-pfor-postings` (2026-10-07) we already observed that
every byte saved in a posting is a byte the scanner does not re-touch.

Two prior techniques pointed toward a stronger encoder:

1. **Locality reorder** (ADR-352, 2026-11-02) rewrites the vector slab so
   that IDs inside a cluster live in a contiguous address range after a
   Hilbert or projection reorder. That compaction turns in-posting IDs
   from a random u32 scatter into a *monotone, near-arithmetic* sequence.
2. **Delta-of-delta** (D2), used in Gorilla, InfluxDB, Prometheus and
   several time-series DBs since ~2015, encodes second differences of a
   monotone sequence. On an arithmetic progression DoD ≡ 0, folding every
   step after the second into a single zigzag-varint byte.

Nobody in the ruvector nightlies so far has composed these two
techniques against a shared IVF layout and measured bytes/ID + decode
throughput head-to-head. This nightly does.

## Decision

Introduce an optional posting-list encoder, `D2Varint`, that writes:

  * `varint( n )` — posting length,
  * `zz_varint( ids[0] )` — first ID,
  * `zz_varint( ids[1] − ids[0] )` — first delta,
  * `zz_varint( (ids[i] − ids[i-1]) − (ids[i-1] − ids[i-2]) )` for i ≥ 2.

Expose this behind a `PostingCodec` trait alongside `RawU32` (baseline)
and `VarintDelta` (single-delta PFOR-lite) so the probe kernel can swap
backends per cluster or globally.

### Design properties

  * No `unsafe`, no SIMD intrinsics. The win is pure geometry.
  * Encoder is streaming; decoder keeps only two i64 registers of state.
  * Codec selection is per posting (store a 2-bit tag in the IVF header),
    so clusters whose IDs fit the `varint-delta` regime do not pay D2's
    extra +1 instruction per element.
  * File-format compatible: both encoders are self-describing by their
    first `varint(n)` and can live side by side.

## Consequences

### Measured (see research doc §Results)

  * **Uniform-random ID assignment (no reorder):** D2 ties `VarintDelta`
    (both 1.94–2.61 B/ID across 1K/4K/16K clusters). No regression,
    no win — the DoD distribution has nonzero variance and buys nothing.
  * **Locality-reordered, stride≈1 slabs:** both encoders hit the 1 B/ID
    varint floor. D2 again ties. Correct outcome — DoD and single delta
    both already compress to one byte.
  * **Sparse-ID AP regime (stride ≥ 1000):** D2 wins.
    - stride=1000, jitter=5: **1.01 B/ID** vs varint-delta's 2.00 B/ID
      → **1.99x** smaller, **3.97x** vs raw.
    - stride=10000, jitter=5: **1.01 B/ID** vs varint-delta's 3.00 B/ID
      → **2.98x** smaller, **3.97x** vs raw.
    - stride=100000, jitter=50: 1.14 B/ID vs 3.00 B/ID → **2.63x**.
  * Decode throughput within 5% of `VarintDelta` on all configs (D2 does
    one extra integer add and one extra register update per element).

### Honest limits

  * D2 is only a win when the common delta exceeds 128 (the single-byte
    varint range). Dense-slab IVF after locality reorder does **not** hit
    that regime — adopt `VarintDelta` there, not D2.
  * D2 shines in multi-tenant or sharded-ID layouts where the 32-bit ID
    namespace is sparse within each posting (common in hybrid search,
    federated indices, or when IDs are hash-prefix-sharded).
  * Benchmarks are synthetic. Real SIFT1M / LAION / Cohere runs are the
    next step.

### Risks

  * Two codecs in the file format widen the test matrix. Mitigation: the
    `PostingCodec` trait is behind a 2-bit per-posting tag; an
    all-`VarintDelta` index is byte-identical to today's PFOR-lite output
    so the migration is monotone.
  * Streaming decode means random-access into a posting requires a scan
    from the posting start. IVF probe kernels scan postings linearly
    anyway, so this is not a regression.

## Alternatives considered

  * **Keep `VarintDelta` everywhere.** Simpler, no new code paths. Loses
    the 1.6–3x compression win on sparse-ID postings.
  * **Elias-Fano** (2026-10-26 nightly). Tighter for very long monotone
    postings but higher per-posting overhead and no DoD advantage for
    small postings. Complementary; expose as a third codec in a future
    ADR after measuring on SIFT1M.
  * **StreamVByte** (2026-10-29 nightly on HNSW adjacency). 1 B/ID floor
    with SIMD decode, but no DoD compaction — loses to D2 in the
    stride ≥ 1000 regime.
  * **Lift to 2x4-bit nibbles.** Smaller constant factor but no
    asymptotic improvement over single-delta varint on AP data.

## Open questions / next iterations

  * Add a hybrid codec that picks D2 or VarintDelta per posting based on
    the stride distribution of its first K elements.
  * SIMD DoD decode (AVX2 / NEON) — compare against StreamVByte's
    SIMD-friendly single-delta kernel.
  * Fold into ADR-352's locality reorder so bench runs the real pipeline
    (reorder → encode → probe) on SIFT1M.
