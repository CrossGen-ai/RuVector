# ruvector-pq-fastscan-lut16

**PQ FastScan with 4-bit LUT16 packed codes and portable SIMD byte-shuffle scan.**

Three trait-swappable Product Quantization (PQ) Asymmetric Distance
Computation (ADC) backends behind a single [`PqAdc`] trait:

| Backend | Codebook | Code size | Scan layout |
|---|---|---|---|
| `ScalarPqAdc` | ks=256 | 1 byte / sub-quantizer / vector | row-major, f32 LUT |
| `Lut16ScalarFastscan` | ks=16 | 4 bits / sub-quantizer / vector | row-major, u8 LUT |
| `Lut16BlockedFastscan` | ks=16 | 4 bits / sub-quantizer / vector | block-transposed 32-wide, u8 LUT |

The blocked variant reshapes the code table so each subquantizer's block-column
(16 bytes = 32 4-bit codes) can be looked up by a 16-lane byte shuffle
(`VPSHUFB` on x86-64, `TBL` on aarch64) against its per-query u8 LUT. This is
the Faiss "fastscan block4" / SCANN "LUT16" layout.

## References

- Jegou et al. "Product Quantization for Nearest Neighbor Search" (2011).
- Andre et al. "Cache locality is not enough: high-performance nearest neighbor
  search with product quantization fast scan" (VLDB 2015).
- Guo et al. "Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization" (ICML 2020) -- SCANN LUT16.

## Run

```
cargo build --release -p ruvector-pq-fastscan-lut16
cargo test  --release -p ruvector-pq-fastscan-lut16
cargo run   --release -p ruvector-pq-fastscan-lut16 --bin fastscan-benchmark
```

## Design notes

- The u8 LUT scales per-sub-quantizer contributions so `sum_over_m(u8) <= 255`,
  giving u16 accumulators plenty of headroom.
- `BLOCK = 32` mirrors Faiss so 16 low nibbles + 16 high nibbles cover the
  block via two 16-lane shuffles.
- Pure safe Rust, `#![deny(unsafe_code)]`. Autovectorization does the SIMD;
  hand-written NEON/AVX2 intrinsics are a future backend behind the same
  trait.
