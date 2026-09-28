# ADR-350: Adaptive Bit-Budget Product Quantization (AB-PQ)

## Status

Accepted (research crate). New, opt-in module `crates/ruvector-abpq`.
Nightly research 2026-09-28. Not wired into `ruvector-pq-search` or
`ruvector-cascade-adc` yet — that lands in follow-up ADRs once the PQx4
LUT-16 layout with mixed strides is designed (see "Consequences").

## Context

`ruvector` ships several PQ-family quantizers: `ruvector-pq-search`,
`ruvector-pq-fastscan-lut16`, `ruvector-cascade-adc`, `ruvector-opq-rotation`,
`ruvector-fused-rabitq-residual`. All of them allocate a **uniform** number
of bits per subquantizer — the standard PQ layout since Jégou 2011.

For anisotropic distributions — which is what real embeddings look like
after PCA, OPQ, or on natural text/image encoders — uniform allocation
wastes bits on low-variance subvectors and starves high-variance ones. The
information-theoretic fix (reverse water-filling, Cover & Thomas ch. 10) is
textbook, but no popular vector DB (Milvus, FAISS, Qdrant, LanceDB,
Weaviate) currently exposes it because of the storage / SIMD-scan
awkwardness that variable bit widths create.

We want a small, self-contained crate that (a) demonstrates the win with
real numbers, (b) leaves a clean trait boundary for integration into the
existing PQ paths, and (c) doesn't regress scan-time performance for equal
code-length.

## Decision

Introduce `ruvector-abpq` (new workspace crate) with:

1. **`water_fill::allocate_bits(variances, total, b_min, b_max)`** — an
   O(B log m) greedy that returns the integer bit assignment minimizing
   `Σ v_i · 2^(-2 b_i)` under `Σ b_i = B, b_min ≤ b_i ≤ b_max`. The
   marginal-gain-decreasing property makes the greedy provably optimal.
2. **`AbPq::train(cfg, data)`** — per-subvector variance → water-filling →
   independent k-means++ codebooks (2^b_i entries each).
3. **`pack::write_bits` / `read_bits`** — 2–8 bit bit-packed codes; every
   field crosses at most one byte boundary; branchless read.
4. **`AbPq::build_adc_table(q)` + `adc_distance(code, table)`** —
   asymmetric distance computation with per-subquantizer block offsets in
   a single flat LUT.

Defaults: `b_min = 2`, `b_max = 8`, k-means iters = 25.

## Consequences

**Positive**

- On synthetic anisotropic data (n=5000, d=64, m=16, avg 4 bits/sub, same
  8-byte code as uniform PQ-4): **MSE −34.0 %** (1.987 → 1.316) and
  **recall@10 +82 %** (0.172 → 0.313) at scan-time parity.
- Trait shape (`AbPqConfig`, `AbPq`, `AdcTable`) drops in behind the
  existing `PqLike` API with only a code-bytes/bit-offset adaptor.
- Water-filling is O(B log m), amortized to zero against k-means training.
- Bit-packed layout is stable, versioned, and serde-serializable.
- Uniform-fallback is trivial (set `b_min = b_max = b_avg`), so
  regression-safe: worst case AB-PQ ≡ classical PQ.

**Negative**

- Variable bit widths break the LUT-16 fastscan trick out of the box.
  Recovery plan: bucket subquantizers by bits, run same-bit groups through
  the existing `ruvector-pq-fastscan-lut16` path, and score the tail
  scalar. Deferred to a follow-up ADR.
- ADC scan is ~2.6 % slower than tight uniform PQ ADC because of variable
  bit-shift extraction. Expected to disappear once same-bit blocks use
  the SIMD LUT path.
- High-variance slots that hit `b_max = 8` on extremely skewed data
  strand budget on lower slots. Raising to `b_max = 10` needs a u16 code
  field. Not fixed here.
- Retraining changes variances → codes must be re-encoded. Same failure
  mode as OPQ; document and version the codebook.

## Alternatives considered

1. **Uniform PQ (baseline)**. Simpler, SIMD-friendly, but leaves the 82 %
   recall gap on the table for typical embedding distributions.
2. **OPQ rotation only**. Solves *which* dimensions land in which
   subquantizer, but still allocates uniform bits within the result. AB-PQ
   composes on top of OPQ; the two are complementary.
3. **RaBitQ (1-bit)**. Very fast scan, but code length is not tunable
   per-subvector; AB-PQ operates at a different code-length regime
   (~8–16 bytes/vec vs RaBitQ's ~4 bytes/vec).
4. **Additive quantization / tree quantization**. Higher accuracy per
   byte but training is 10–100× slower and codes can't be pre-packed for
   LUT scan.
5. **Continuous rate-distortion allocation**. The continuous solution
   gives fractional bits which don't correspond to any k-means codebook;
   the integer greedy is the correct discretization.

## References

- Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, TPAMI 2011.
- Ge, He, Ke, Sun, *Optimized Product Quantization*, CVPR 2013.
- Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020 (ScaNN).
- Gao & Long, *RaBitQ*, SIGMOD 2024.
- Cover & Thomas, *Elements of Information Theory* (2nd ed.), ch. 10.
- Research doc: `docs/research/nightly/2026-09-28-adaptive-bit-budget-pq/README.md`.
- Implementation: `crates/ruvector-abpq/`.
