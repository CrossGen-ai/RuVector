# ADR-348: PQ FastScan with LUT16 (4-bit packed codes + byte-shuffle scan)

- **Status**: Proposed (nightly-research prototype)
- **Date**: 2026-09-23
- **Related**: ADR-347 (opq-rotation), earlier PQ-ADC work in `ruvector-pq-search`

## Context

The current PQ-ADC path in ruvector (`ruvector-pq-search`, `ruvector-cascade-adc`)
uses classical Jegou 2011 Product Quantization: each vector is encoded as `m`
one-byte codes selecting among `ks = 256` centroids per sub-quantizer.
Query-time distance computation builds an `[m][256] f32` LUT and scans the
DB row-by-row summing `m` f32 lookups per vector.

Two independent bottlenecks show up under profiling:

1. **Cache footprint.** At `m = 16` the LUT alone is `16 * 256 * 4 = 16 KiB`;
   the code table is `n * 16` bytes, which for `n = 1M` is 16 MiB and does not
   fit in L2. The scan is DRAM-bandwidth bound.

2. **No SIMD gather.** A 256-entry LUT cannot be held in vector registers, so
   each code turns into a per-lane f32 gather. AVX2's `vpgatherd` is far
   slower than register-resident lookup on real workloads.

Faiss (`fast_scan`), SCANN (`LUT16`), and the RaBitQ family have all moved to
**4-bit codes with 16-entry LUTs**. A 16-byte LUT fits in a single NEON `q`
register (or an AVX2 `xmm`) and one `TBL` / `VPSHUFB` looks up 16 codes in a
single cycle. Codes shrink 2x. Faiss reports 5-10x scan speedups on x86.

Ruvector has RaBitQ (`ruvector-rabitq`, `ruvector-fused-rabitq-residual`) and
byte-code PQ (`ruvector-pq-search`) but no LUT16 FastScan crate. This ADR adds
one.

## Decision

Add `crates/ruvector-pq-fastscan-lut16/` containing three trait-swappable
backends behind a single `PqAdc` trait:

| Backend | Codebook | Code size | Layout |
|---|---|---|---|
| `ScalarPqAdc` | ks=256, m | 8 bits per sub-quantizer per vector | row-major, f32 LUT |
| `Lut16ScalarFastscan` | ks=16, m | 4 bits per sub-quantizer per vector | row-major packed, u8 LUT |
| `Lut16BlockedFastscan` | ks=16, m | 4 bits per sub-quantizer per vector | block-transposed (BLOCK=32), u8 LUT |

The blocked layout stores, for each 32-vector block and each sub-quantizer,
16 contiguous bytes: 32 4-bit codes with lanes 0..16 in the low nibble and
lanes 16..32 in the high nibble. This is the Faiss "fastscan block4" layout
and is the input shape a NEON `TBL` / AVX2 `VPSHUFB` byte-shuffle expects.

The u8 LUT scales per-sub-quantizer contributions to `[0, 255/m]` so a u16
accumulator over `m` sub-quantizers never overflows, and a single `scale`
factor recovers the true squared-L2 distance.

## Consequences

**Real numbers on Apple M-series, `--release`, isotropic Gaussian
n=50000 d=64 m=8 nq=200:**

```
ScalarPqAdc            (ks=256, 8-bit)   R@100 of GT@10 = 0.537   0.740 ms/query   390 KiB
Lut16ScalarFastscan    (ks=16,  4-bit)   R@100 of GT@10 = 0.196   0.296 ms/query   195 KiB
Lut16BlockedFastscan   (ks=16,  4-bit)   R@100 of GT@10 = 0.196   0.321 ms/query   195 KiB
```

- **Scan speed:** 2.5x faster than 8-bit baseline (DRAM-bandwidth win from 2x
  smaller code table, u8 LUT fits in cache, u16 accumulators avoid f32
  arithmetic).
- **Code storage:** exactly 2x smaller as designed.
- **Recall:** drops from 0.537 to 0.196 for the same `m` because `ks = 16`
  centroids per subspace has 4x less representational power than `ks = 256`.
  In production this is recovered by **doubling `m`** (still fits within a
  16-byte block column) plus **OPQ rotation** (ADR-347). At `m = 16` and OPQ,
  LUT16 typically matches classical PQ recall in the literature.
- **Blocked vs scalar scan:** the blocked layout is not yet faster than the
  scalar packed layout in this PoC because we rely on autovectorization; the
  compiler on `aarch64` does not emit `TBL` for the current pattern. The
  layout is prepared for a future intrinsics-backed backend that will land
  under the same trait.

**Alternatives considered:**

- **8-bit code + AVX-512 VPERMI2B (SCANN style).** Requires x86-64 only,
  which ruvector does not want to hard-fork on. LUT16 is portable.
- **Half-precision (f16) LUT.** Cuts LUT footprint 2x but does not shrink
  codes, so it does not remove the DRAM bottleneck. Complementary, not
  substitute.
- **RaBitQ (already in tree).** 1 bit per dim with a rotation; even smaller
  than LUT16 but has a different accuracy profile (bounded relative error
  vs bounded absolute error). Both belong in the toolbox.

**Follow-ups:**

- Feature-gated NEON `vqtbl1q_u8` and AVX2 `_mm256_shuffle_epi8` backends,
  same trait.
- Compose with OPQ (ADR-347) and RaBitQ residuals for a two-stage 4-bit
  cascade -- expected recall parity with 8-bit at 2x speed.
- Wire the blocked layout into `ruvector-pq-search` behind a feature flag.

## Layout in-tree

- `crates/ruvector-pq-fastscan-lut16/` -- new crate, no unsafe, no external
  BLAS/FFI.
- `docs/research/nightly/2026-09-23-pq-fastscan-lut16/README.md` -- research
  doc + full benchmark methodology.
