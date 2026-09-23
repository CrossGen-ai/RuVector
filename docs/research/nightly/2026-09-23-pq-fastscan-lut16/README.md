# PQ FastScan with LUT16: 4-bit Packed Codes and Portable Byte-Shuffle Scan

**Nightly research** · 2026-09-23 · slug: `pq-fastscan-lut16`

## Abstract

Classical Product Quantization (Jegou et al., 2011) uses one byte per
sub-quantizer per vector with a 256-entry f32 lookup table (LUT) built per
query. Modern vector search engines (Faiss FastScan, SCANN LUT16) instead use
**4-bit codes with 16-entry u8 LUTs** because a 16-byte LUT fits in one SIMD
register and a single byte-shuffle (`VPSHUFB` / `TBL`) resolves 16 codes per
cycle. We build a portable-Rust prototype of the LUT16 FastScan technique in a
new crate, `ruvector-pq-fastscan-lut16`, exposing three trait-swappable
backends: `ScalarPqAdc` (baseline 8-bit), `Lut16ScalarFastscan` (4-bit codes,
row-major), and `Lut16BlockedFastscan` (4-bit codes, 32-wide block-transposed
layout). On isotropic-Gaussian n=50k d=64 m=8 the LUT16 variants **cut scan
latency 2.5x** (0.74 → 0.30 ms/q) and **halve code storage** (390 → 195 KiB)
at the cost of recall proportional to the ks=16 vs ks=256 codebook capacity
ratio -- a well-understood trade normally recovered by doubling `m` and
applying OPQ (see ADR-347).

## State of the Art

- **Jegou, Douze, Schmid, 2011** -- classical PQ + ADC. `m` byte codes,
  256-entry f32 LUT per sub-quantizer, `O(m)` adds per DB vector.
- **Andre, Kermarrec, Le Scouarnec, VLDB 2015** -- "PQ Fast Scan":
  observed that the main cost is table lookups, not arithmetic. Introduced
  4-bit codes + `VPSHUFB` on x86 for 4-8x speedup. Blocked layout with 32
  vectors per block became the standard.
- **Guo, Sun, Lindgren et al., ICML 2020 (SCANN)** -- combined LUT16 with
  anisotropic loss for MIPS on billion-scale corpora at Google. LUT16 is core.
- **Douze et al., Faiss `IndexPQFastScan`** -- production LUT16
  implementation with NEON, AVX2, AVX-512 backends. Reference for the
  block-32 4-bit layout used here.
- **Gao, Long, 2024 -- RaBitQ.** 1-bit-per-dim with theoretical error bounds,
  a step beyond LUT16 on recall/byte but with a different accuracy profile.
  Already in the ruvector tree (`ruvector-rabitq`), complementary to LUT16.

Ruvector already has classical PQ (`ruvector-pq-search`), OPQ rotation
(`ruvector-opq-rotation`, ADR-347), and RaBitQ. The one missing ingredient in
the tree until this ADR was LUT16 / FastScan itself.

## Proposed design

A single `PqAdc` trait:

```rust
pub trait PqAdc {
    fn len(&self) -> usize;
    fn code_bytes(&self) -> usize;
    fn m(&self) -> usize;
    fn ks(&self) -> usize;
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>;
}
```

Three backends behind it:

| Backend | Codebook | Storage | LUT type | Scan |
|---|---|---|---|---|
| `ScalarPqAdc` | ks=256, m | `n*m` bytes | f32 [m][256] | row-major, per-vec loop |
| `Lut16ScalarFastscan` | ks=16, m | `n*m/2` bytes | u8 [m][16] | row-major, per-vec loop |
| `Lut16BlockedFastscan` | ks=16, m | `n/32 * m * 16` bytes | u8 [m][16] | 32-wide block, per-sub-q byte-shuffle |

### The block layout in detail

For BLOCK=32 vectors and m sub-quantizers, one block stores `m * 16` bytes:

```
block b:
  sub[0] = 16 bytes:
    byte[t=0..16]:
      low  nibble  = code(vector t,      sub-quantizer 0)
      high nibble  = code(vector t + 16, sub-quantizer 0)
  sub[1] = 16 bytes: ... same for sub-quantizer 1
  ...
```

The scan for one block, one sub-quantizer is exactly the operation a NEON
`vqtbl1q_u8` (or AVX2 `_mm256_shuffle_epi8`) executes:

```
vec<u8, 16> low_codes  = sub[sq];              // 16 lanes = 16 4-bit codes
vec<u8, 16> hi_codes   = sub[sq] >> 4;         // 16 lanes = other 16 codes
vec<u8, 16> low_lookup = shuffle(lut[sq], low_codes);  // per-vec u8 distance
vec<u8, 16> hi_lookup  = shuffle(lut[sq], hi_codes);
acc_low  = wrap_add_u16(acc_low,  low_lookup);
acc_high = wrap_add_u16(acc_high, hi_lookup);
```

This is what Faiss `IndexPQFastScan` compiles to. Our safe-Rust portable
version writes the same computation as a nested loop; a NEON/AVX2 intrinsics
backend behind the same trait is left as a follow-up.

### Distance quantization

Per-query, per-sub-quantizer, per-centroid squared-L2 distances are
quantized to u8 with a **shared max** so a u16 accumulator over `m`
sub-quantizers can never overflow:

```
t[sq][c]   = ||q_sub_sq - centroid[sq][c]||^2
t_max      = max over all (sq, c) of t[sq][c]
per_sq_max = 255 / m
lut[sq][c] = round(t[sq][c] / t_max * per_sq_max)  // u8 in [0, 255/m]
scale      = t_max * m / 255                        // f32 recovery factor
```

Reconstructed distance for candidate `i`:
`d(q, x_i) ~= scale * sum_over_sq(lut[sq][code_i[sq]])`.

## Implementation notes

- `#![deny(unsafe_code)]` throughout. No BLAS, no FFI, no nightly features.
- k-means training: plain Lloyd with random-row seeding, 20 iterations.
  Empty clusters re-seeded to a random point.
- No probing structure (IVF, HNSW) -- this crate is deliberately just the
  scan primitive, to be composed by an outer index. `ruvector-pq-search`
  and `ruvector-cascade-adc` already provide those.
- BLOCK=32 chosen because 32 * 4 bits = 16 bytes = one NEON `q` register or
  one AVX2 `xmm`.
- `Cargo.toml` is workspace-inherited so ruvector's `--workspace` sweep
  compiles it.

## Benchmark methodology

- **Hardware**: Apple M-series (aarch64), stable `rustc 1.89.0`, release
  profile with LTO off (workspace default).
- **Data**: isotropic zero-mean Gaussian via CLT (sum of 12 uniform draws
  minus 6). n=50000 database, nq=200 held-out queries generated with a
  different seed.
- **Metric**: R@100 of ground-truth top-10. Ground truth computed by exact
  squared-L2 brute force. This is the standard ANN metric used by Faiss and
  SCANN benchmarks.
- **Parameters**: d=64, m=8, ds=8. All three backends trained with the same
  seed and iteration count (20). LUT16 variants use ks=16; baseline uses
  ks=256.
- **Timing**: 3-query warmup, then `Instant::now()` around the 200-query
  sweep, wall-clock. Per-query = total / nq.

## Results

Real numbers, `cargo run --release --bin fastscan-benchmark`:

```
d=64 n=50000 nq=200 gt_k=10 m=8

Training ScalarPqAdc          (ks=256, m=8)   trained in 3.48s
Training Lut16ScalarFastscan  (ks=16,  m=8)   trained in 0.26s
Training Lut16BlockedFastscan (ks=16,  m=8)   trained in 0.25s

  ScalarPqAdc          (ks=256, 8-bit)  R@100 of GT@10 = 0.537   0.740 ms/query   codes = 390 KiB
  Lut16ScalarFastscan  (ks=16,  4-bit)  R@100 of GT@10 = 0.196   0.296 ms/query   codes = 195 KiB
  Lut16BlockedFastscan (ks=16,  4-bit)  R@100 of GT@10 = 0.196   0.322 ms/query   codes = 195 KiB

scan speedup: scalar-fs 2.50x   blocked-fs 2.31x
code size:    390 -> 195 KiB (2.00x smaller)
```

- **2.5x scan speedup** from the u8 LUT + 2x smaller code table (better cache
  residency, u16 accumulator instead of f32).
- **Exactly 2x code compression**, as designed.
- **Recall drops** from 0.537 to 0.196 at the same `m`. This is the ks=16 vs
  ks=256 codebook-capacity delta and is the price for the shuffle-friendly
  4-bit layout. In production this is closed by (a) doubling `m`, which
  LUT16's 4-bit encoding pays for out of the storage savings, and (b) OPQ
  rotation (ADR-347) which the same trait can wear.

## How it works (walkthrough)

1. **Train.** For each of the m sub-spaces, run k-means with ks centroids
   over the projected data. Store the `m * ks * ds` codebook.
2. **Encode.** For each vector, for each sub-quantizer, find the nearest
   centroid id; emit ks<=16 → 4-bit id or ks<=256 → 8-bit id.
3. **Pack (LUT16).** Two 4-bit codes per byte. For the blocked variant,
   permute so lanes 0..16 fill the low nibbles and lanes 16..32 fill the high
   nibbles of a 16-byte column-per-sub-quantizer.
4. **Query LUT.** For an incoming query, split into m sub-vectors of size
   ds. Compute distance from each sub-vector to each of ks centroids; that's
   an `m * ks` table. Quantize to u8 with a shared max so u16 accumulators
   suffice.
5. **Scan.** For each block, for each sub-quantizer, shuffle the 16-byte LUT
   through the 16 code bytes (twice, once for low nibbles once for high) to
   get u8 distances. Add into a `[u16; 32]` block accumulator.
6. **Reconstruct.** Multiply the u16 accumulator by the recovery scale to get
   an approximate squared-L2 distance, then top-k.

## Practical failure modes

- **ks=16 loses fine detail.** Under high intrinsic dimensionality per
  sub-space, 16 centroids can be far from any single sub-vector. Mitigation:
  double `m` (still within block width), pre-rotate with OPQ (ADR-347).
- **u8 LUT saturation.** A single very-large centroid distance eats the whole
  u8 range and drives the rest to zero. Real Faiss uses per-sub-quantizer
  scale + a rerank stage. This PoC uses a global scale for simplicity;
  hot-swap in a per-sq scale later.
- **Portable scan is slower than intrinsics.** As the numbers show,
  autovectorization on aarch64 does not emit `TBL` for the current pattern
  (blocked and scalar run at the same speed). This is the expected next-step
  gain from a `#[cfg(target_arch = "aarch64")]` intrinsics backend.
- **Small n.** The 2.5x speedup is a memory-bandwidth win that shrinks as n
  drops below L2 residency for both variants. At n=1M the gap is expected to
  widen further.

## What to improve next (roadmap)

1. **Intrinsics backends.** NEON `vqtbl1q_u8`, AVX2 `_mm256_shuffle_epi8`,
   AVX-512 `vpermb`. Same trait, gated by `target_feature`. Expected +2-3x
   over the portable scan.
2. **Per-sub-quantizer scales** to avoid one large centroid crushing the
   whole LUT.
3. **Compose with OPQ (ADR-347)** for recall recovery. Rotate the query and
   codebook consistently, train with alternating Kabsch, expect recall to
   approach the ks=256 baseline while keeping the 2x smaller codes and 2.5x
   scan speed.
4. **Blocked + rerank cascade.** Return the top-`nprobe` from the LUT16 scan,
   then exact-rescore in f32. Standard Faiss / SCANN pattern. Trivially
   layered on top of `PqAdc::search`.
5. **Wire into `ruvector-pq-search`** as an alternate scan backend behind a
   feature flag.

## Production crate layout proposal

```
crates/ruvector-pq-fastscan-lut16/
  Cargo.toml
  README.md
  src/
    lib.rs        # PqAdc trait, three portable backends, kmeans, encoding
    main.rs       # fastscan-benchmark binary
  benches/        # criterion later
```

For upstreaming into `ruvector-pq-search`:

- Add a `ScanBackend` enum: `Byte8`, `Lut16Row`, `Lut16Block`.
- Move the `PqAdc` trait into `ruvector-core` under `pq::AdcBackend`.
- Wire NEON/AVX2 backends behind cargo features `simd-neon`, `simd-avx2`,
  `simd-avx512`.

## References

- Jegou, Douze, Schmid. "Product Quantization for Nearest Neighbor Search."
  IEEE TPAMI, 2011.
- Andre, Kermarrec, Le Scouarnec. "Cache locality is not enough:
  high-performance nearest neighbor search with product quantization fast
  scan." VLDB 2015.
- Guo et al. "Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization." ICML 2020 (SCANN).
- Douze et al. "The Faiss library." arXiv:2401.08281, 2024.
- Gao, Long. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
  Error Bound." SIGMOD 2024.
- ruvector ADR-347 -- OPQ rotation (2026-09-22).
- ruvector `ruvector-pq-search`, `ruvector-cascade-adc`, `ruvector-rabitq`.
