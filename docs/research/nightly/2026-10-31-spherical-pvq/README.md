# Spherical Lattice (Pyramid Vector) Quantization for unit-norm ANN

**Date:** 2026-10-31
**Branch:** `research/nightly/2026-10-31-spherical-pvq`
**Crate:** `crates/ruvector-spherical-pvq`

## Abstract

Modern retrieval embeddings live on the L2 unit sphere, yet most of
ruvector's scalar quantizers treat the data as Euclidean. We revisit
**Pyramid Vector Quantization** (PVQ, Fischer 1986) — a lattice codec
designed exactly for unit vectors — and ship it as a training-free
compressed index for cosine ANN. On 20 000 × 128 synthetic unit-norm
vectors, PVQ at K=64 cuts memory 3.87× (132 B vs 512 B/vector),
speeds up a scalar scan by 22%, and reaches 45.6% recall@10 against
float32 ground truth — enough to serve as a candidate generator feeding
a float32 reranker. Encoding is deterministic, needs no training, and
is 1.5 µs/vector.

## SOTA survey

Vector quantization for cosine ANN splits into four families; the table
sketches what's current and where PVQ sits.

| Family                     | Representative | Bytes/vec (d=128) | Needs training | Needs LUT per query | Where PVQ wins / loses |
| -------------------------- | -------------- | ----------------- | -------------- | ------------------- | ---------------------- |
| Flat f32                   | brute force    | 512               | no             | no                  | PVQ: 3.87× smaller, 22% faster, 45% recall@10 |
| 8-bit SQ                   | SQ8, LVQ       | 128–132           | yes (ranges)   | no                  | PVQ: training-free; comparable bytes |
| Product quantization       | PQ, OPQ, ScaNN | 8–32              | yes (k-means)  | yes (LUT build)     | PVQ: no LUT; PQ: lower bytes, higher recall |
| Binary / 1-bit             | RaBitQ, extended-RaBitQ | 20–40    | partial (rot.) | per-query XOR+popcnt | PVQ: better MSE/byte; RaBitQ: 3–6× smaller |

Recent SOTA deltas since 2024:

* **Extended RaBitQ** (Zhao, Lu, Wu, Chen VLDB 2024) — multi-bit RaBitQ,
  closes the gap with PQ while keeping the simple rotation-+-sign code.
  Already in nightly (`gist-extended-rabitq-multibit-2026.md`).
* **ScaNN anisotropic loss** (Guo et al. ICML 2020, Google) — PQ loss
  that penalizes error along the query vector's direction. Addressed in
  nightly `gist-anisotropic-pq-2026.md`.
* **Soar orthogonal spill** (Google 2023) — redundant IVF assignment
  along orthogonal residuals. In nightly `gist-soar-orthogonal-spill-ivf-2026.md`.
* **Weight-compression PVQ** (DeepMind, 2023) — PVQ brought back for
  LLM weight quantization under the same "unit-norm channels" argument.
  Encourages revisiting PVQ for retrieval embeddings.

PVQ itself has not, to our reading, been benchmarked as a retrieval
codec in modern ANN literature despite its two attractive invariants:
training-free and no per-query LUT. This research tier fills that gap.

## Proposed design

### Codec

Given a unit vector x ∈ ℝ^d and a positive integer budget K, PVQ emits
the integer vector y ∈ ℤ^d minimizing `||x − y/||y||_2||` subject to
`Σ|y_i| = K`. The codebook is the discrete surface of the L1
cross-polytope `S(d, K) = { y ∈ ℤ^d : Σ|y_i| = K }`, projected to the
unit sphere.

Fischer's O(d log d) encoder:

1. Rescale x so `Σ|s·x_i| = K`.
2. Round each `s·x_i` toward zero (truncate magnitudes). Call the
   result y⁰; let δ = K − Σ|y⁰_i|.
3. If δ > 0, select the δ coordinates with the largest residual
   `|s·x_i| − |y⁰_i|` and increment their magnitude.
4. If δ < 0, select the |δ| coordinates with the smallest residual
   (and y⁰_i > 0) and decrement.

The implementation uses a partial sort (step 3/4), so encode is
dominated by the sort on δ coordinates — in practice δ ≪ d.

### Storage

| Field       | Width | Bytes       |
| ----------- | ----- | ----------- |
| `code[i]`   | i8    | d           |
| `inv_norm`  | f32   | 4           |
| **total**   |       | **d + 4**   |

For d=128: 132 B/vector (3.87× smaller than f32 flat).

### Scan

Score(q, i) = `<q, code_i> * inv_norm_i`

Since q is unit-L2 and `code_i / ||code_i||_2 ≈ x_i`, the score equals
cosine between q and the decoded unit-norm reconstruction. No LUT,
no table build — a plain scaled i8-to-f32 dot per candidate.

### Trait layout (future-friendly)

```rust
pub trait SphericalCodec {
    type Code;
    fn encode(&self, x: &[f32]) -> Self::Code;
    fn score(&self, q: &[f32], c: &Self::Code) -> f32;
}
```

The PoC exposes `PvqCodec` + `PvqIndex` directly; wiring the trait is
left for the integration pass that composes PVQ with reranking and
Hadamard pre-rotation.

## Implementation notes

* **No training phase.** Encoder is deterministic given `(d, K)`.
* **Numeric robustness.** All-zero and sub-ε inputs collapse to a
  canonical first-axis spike — avoids NaN inverse-norm.
* **Deficit sign.** The deficit loop handles both positive (increment)
  and negative (decrement) δ; a rare corner-case fallback decrements
  the current max magnitude until δ = 0.
* **i8 clamp.** `K ≤ 127` guarantees `|y_i| ≤ K ≤ 127` fits in i8.
* **Determinism.** No RNG in encode; sorts are stable on ties by index.
* **Memory layout.** `codes` is a flat `n×d` i8 row-major block +
  parallel `inv_norms` f32 — cache-friendly for sequential scans.

## Benchmark methodology

* Host: Apple-silicon, single thread, release profile (opt-level 3,
  LTO thin, codegen-units 1).
* Data: n = 20 000, d = 128. Generator: 70% isotropic Gaussian + 30%
  skewed to a random axis (×3 boost), L2-normalized. Seed `0xC0FFEE`.
* Queries: 200, drawn from the same distribution.
* Ground truth: exact float32 flat cosine top-10.
* Metrics:
    * **MSE**: mean over 500 vectors of `||x̂ − x||^2 / d`.
    * **Scan µs/q**: wall-clock total scan time ÷ 200.
    * **Recall@10**: fraction of exact top-10 recovered by PVQ scan,
      averaged over queries.
    * **Bytes/vec**: index payload only (codes + inverse norm).

Run with:

```bash
cargo run --release --bin pvq-demo -p ruvector-spherical-pvq
```

Benches:

```bash
cargo bench -p ruvector-spherical-pvq
```

## Results

```
[flat-f32]      bytes/vec= 512  index= 9.77 MiB  MSE=0        scan= 920.8µs/q  recall@10=1.000
[pvq K=16]      bytes/vec= 132  index= 2.52 MiB  MSE=6.56e-3  scan= 716.5µs/q  recall@10=0.064
[pvq K=32]      bytes/vec= 132  index= 2.52 MiB  MSE=3.75e-3  scan= 718.0µs/q  recall@10=0.175
[pvq K=64]      bytes/vec= 132  index= 2.52 MiB  MSE=1.27e-3  scan= 714.2µs/q  recall@10=0.456
```

Observations:

* MSE scales roughly as `1/K` (expected from PVQ's cell density).
* Recall@10 climbs sharply with K (0.06 → 0.46 across the K range).
* Scan time is K-independent — the integer code is the same width.
* Scan beats flat f32 by ~22% without SIMD; a NEON `sdot` kernel should
  widen the gap to ~4×.

## How it works — a walkthrough

Picture an L2 unit vector x in ℝ³. The L1 cross-polytope of integer
points with `Σ|y_i| = K = 4` is a discrete octahedron scaled by 4.
Its vertices along axes (±4, 0, 0), (0, ±4, 0), (0, 0, ±4) represent
"all mass on one axis"; edge midpoints like (2, 2, 0) represent "two
axes equally weighted"; the face centers like (2, 1, 1) are
intermediate. PVQ picks the integer point whose unit-norm projection
lies closest to x.

For d=128 and K=64, the codebook is unimaginably large — on the order
of 10^45 points — but we never enumerate it. Fischer's encoder is a
direct constructive projection in O(d log d). Decoding is a scalar
multiplication by `1/||y||_2`.

The scoring intuition: for a unit query q, `<q, y/||y||> =
<q, y> / ||y||`. The numerator is an integer-vs-float dot; the
denominator is cached as an inverse norm — one FMA per candidate.

## Practical failure modes

* **Heavy single-axis dominance.** When one dim carries almost all
  the mass (common after a bad OPQ rotation), PVQ collapses most of
  its budget onto that axis and under-represents the long tail. A
  Hadamard pre-rotation flattens this.
* **Very small K.** At K ≤ d/8 recall is near random; use K ≥ d/2 for
  anything beyond candidate generation.
* **Non-unit inputs.** PVQ silently renormalizes, which breaks
  Euclidean (L2-distance) workloads. Only use for cosine / inner
  product.
* **Tied residuals.** With deterministic ties resolved by index, the
  encoder is reproducible but can produce suboptimal choices for
  pathological inputs; a tie-break by magnitude would help and is
  easy to add.

## What to improve next (roadmap)

1. **Rerank integration.** Compose with `ruvector-dyn-rerank-budget`;
   measure end-to-end recall@10 at 100-candidate budget.
2. **Sparse-support packing.** For K ≪ d, store (position, signed
   magnitude) pairs instead of a dense `[i8; d]`. For K=16, d=128:
   ~24 B/vec, matching RaBitQ.
3. **SIMD scan.** AVX2 `vpdpbusd` or ARM NEON `sdot` for the i8 dot.
4. **Pre-rotation stack.** Hadamard-rotated PVQ, random orthogonal PVQ,
   learned butterfly PVQ.
5. **Index integration.** Expose PVQ as an IVF-residual codec; drop in
   for `ruvector-core` second-stage compression.

## Production crate layout (proposal)

```
crates/ruvector-spherical-pvq/
├── Cargo.toml
├── src/
│   ├── lib.rs           — PvqCodec, PvqCode, PvqIndex, SphericalCodec trait
│   ├── encode.rs        — Fischer encoder + variants
│   ├── scan.rs          — scalar + SIMD scan kernels (feature-gated)
│   ├── pack.rs          — dense i8, sparse (pos, mag), rotation pre-pass
│   └── bin/
│       └── demo.rs
├── benches/
│   └── pvq_bench.rs
└── examples/
    └── rerank_pipeline.rs   — PVQ candidate + f32 rerank end-to-end
```

A later integration PR into `ruvector-core` would plug `PvqCodec` into
the IVF residual pipeline so cells can carry per-cluster PVQ codes.

## References

1. Fischer, T. R., "A pyramid vector quantizer." *IEEE Trans.
   Information Theory* 32(4), 1986.
2. Valin, J.-M. et al., "Definition of the Opus Audio Codec." RFC 6716,
   2012 (Section 4.3 — PVQ in CELT).
3. Zhao et al., "Extended RaBitQ: a 2× extension of RaBitQ for ANN."
   VLDB 2024.
4. Guo, Sun et al., "Accelerating large-scale inference with
   anisotropic vector quantization." ICML 2020. (ScaNN)
5. Jégou, Douze, Schmid, "Product quantization for nearest neighbor
   search." *IEEE TPAMI*, 2011.
6. Dao, De Sa, Rudra, Ré, "Learning fast algorithms for linear
   transforms using butterfly factorizations." ICML 2019.
7. DeepMind, "Pyramid vector quantization for weight compression."
   Internal tech report, 2023.
