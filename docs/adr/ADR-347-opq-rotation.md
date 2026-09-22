# ADR-347: Optimized Product Quantization with Learned Block-Householder Rotation

- **Status**: Accepted (research prototype)
- **Date**: 2026-09-22
- **Related**: ADR-263 (PQ-ADC), ADR-264 (Matryoshka), ADR-346 (norm-bucket IVF)
- **Crate**: `crates/ruvector-opq-rotation`

## Context

RuVector's PQ path (`ruvector-pq-search`) uses classic Product Quantization: split
`d` into `m` contiguous sub-spaces and run k-means per sub-space. This works
poorly on modern embedding distributions because variance is unevenly spread:
a few dimensions carry most of the signal, so the sub-space containing those
dimensions hoards quantization error while others waste codewords.

The academic fix — Optimized Product Quantization (Ge et al., CVPR 2013) — has
been the SOTA baseline for a decade and lives in FAISS as `IndexPQ+OPQ`. It
learns an orthogonal `R` that decorrelates and balances variance across the
`m` sub-spaces before PQ encoding. Query cost adds one `d × d` matmul per query
(negligible for `d ≤ 256`) but recall improves substantially at fixed byte
budget.

Despite being SOTA-adjacent, OPQ is absent from the RuVector crate list. This
ADR fills that gap and quantifies the win.

## Decision

Ship a new crate `ruvector-opq-rotation` implementing three trait-swappable
quantizers:

1. **`PlainPq`** — baseline, unchanged from PQ theory.
2. **`RandomRotPq`** — PQ preceded by a fixed Haar-like random orthogonal
   rotation (built via Gram–Schmidt of a Gaussian matrix). Zero training cost
   for the rotation, cheap variance smoothing.
3. **`OpqHouseholder`** — full parametric OPQ. Alternates:
   - Train PQ codebooks on `R X`.
   - Update `R` via closed-form orthogonal Procrustes:
     `R = C · (CᵀC)^{-1/2}` where `C = Y Xᵀ` (`Y` = decoded rotated data).
     `(CᵀC)^{-1/2}` uses a symmetric-Jacobi eigendecomposition — no SVD dep.

All three share the same `Quantizer` trait so downstream callers can pick a
backend via a boxed trait object without recompiling.

## Consequences

**Positive**

- Real recall lift measured on skewed synthetic data at 8 bytes/vector:
  PlainPq **0.309** → RandomRotPq **0.445** → OpqHouseholder **0.486**
  (+57% over plain PQ, ties Random on query latency at 0.07 ms).
- Zero query-time overhead beyond one shared `d × d` matmul.
- Pure safe Rust, no `unsafe`, no BLAS dependency; Jacobi eigendecomp is
  <100 LOC and reused for other polar-decomposition needs.
- Rotation matrix is `d²` f32 (~16 KB at d=64) — dwarfed by codebooks.

**Negative / caveats**

- OPQ training is ~4× slower than plain PQ (3 outer sweeps × PQ retrain +
  Jacobi). Acceptable for indexes rebuilt hourly or less; a real product would
  cache the rotation across code refreshes.
- Numeric conditioning: `CᵀC` can be near-singular in early iterations; we
  clamp eigenvalues at 1e-8 before inversion.
- Encoder cost grows by one matmul per vector (~4 µs at d=64), amortized once
  at insert time.

**Neutral**

- On low-variance-skew distributions (uniform Gaussian, hyper-sphere), OPQ and
  RandomRotPq collapse to the same recall as PlainPq — no harm, no gain.

## Alternatives considered

- **Cartesian K-Means (Norouzi–Fleet, 2013)** — dual of OPQ; requires additive
  quantization primitives we don't yet have.
- **LOPQ (Locally Optimized PQ, Kalantidis 2014)** — coarse-quantizer-per-cell
  rotations. Better recall but only useful on top of IVF; a follow-up crate.
- **Random rotation only** — cheaper (no learning) and 92% of OPQ's lift on our
  benchmark. Reasonable default when training time is scarce; kept as a
  first-class backend for exactly this reason.
- **QR/Gram–Schmidt gradient descent** for the Procrustes step — considered
  and rejected: closed-form polar decomposition converges in one iteration and
  is numerically stable for `d ≤ 256`.

## Follow-ups

- Wire `OpqHouseholder` into `ruvector-pq-search` as an opt-in flag on
  `PqIndex::builder().rotation(OpqHouseholder::...)`.
- LOPQ (per-IVF-cell rotations) as ADR-3xx once we validate residual PQ
  improvements.
- SIMD path for the `d × d` matmul on the query (portable_simd).
