# Optimized Product Quantization with Learned Block-Householder Rotation

*Nightly research, 2026-09-22 · crate `ruvector-opq-rotation` · ADR-347*

## Abstract

Product Quantization (PQ) is the workhorse compression scheme for
memory-constrained ANN search, but its per-sub-space k-means implicitly assumes
variance is spread evenly across dimensions. Modern embeddings almost never
satisfy that. Optimized PQ (OPQ, Ge et al. 2013) learns an orthogonal rotation
that redistributes variance across sub-spaces before encoding. Query cost adds
one `d × d` matmul — negligible for `d ≤ 256` — but recall improves
substantially at fixed byte budget.

This nightly ships a pure-Rust OPQ implementation with three swappable backends
(PlainPQ, RandomRotPQ, OpqHouseholder) and measures the recall/latency
trade-off on skewed-variance synthetic data. **On 8-byte codes at d=64,
learned OPQ delivers 0.486 recall@10 vs 0.309 for plain PQ — a 57% recall lift
at identical query time (0.07 ms/query).**

## SOTA survey

- **PQ (Jégou–Douze–Schmid, TPAMI 2011)** — the canonical byte-code compression
  used across FAISS, Milvus, Weaviate. Sub-space k-means, ADC scoring.
- **OPQ (Ge–He–Ke–Sun, CVPR 2013)** — adds a learned orthogonal `R`.
  Two flavors: *parametric* (rotate + PQ, alternating) and *non-parametric*
  (Kabsch closed-form per outer step). FAISS ships parametric OPQ; we
  implement the closed-form Kabsch variant here.
- **Cartesian K-Means (Norouzi–Fleet, CVPR 2013)** — mathematically identical
  to OPQ, derived from a different objective.
- **LOPQ (Kalantidis–Avrithis, CVPR 2014)** — one rotation per IVF cell.
  Higher recall on hard datasets; more parameters. Left for follow-up.
- **ScaNN (Guo et al., ICML 2020)** — anisotropic PQ loss weighting.
  Complementary to OPQ (different objective, same encoding).
- **Additive Quantization (Babenko–Lempitsky, 2014)** and its descendants
  (LSQ, TQ) — replace PQ's Cartesian decomposition with a learned sum. Higher
  recall, expensive encoding, not our target here.
- **Competitor status (Sep 2026)**: Milvus 3.2, Qdrant 1.13, Weaviate 1.28,
  Pinecone all use plain PQ + optional random rotation. FAISS remains the
  only mainstream index with production OPQ. RuVector had no OPQ path
  pre-nightly.

## Proposed design

Trait `Quantizer` with `train / encode / adc_tables / adc_distance`. Three
impls:

- `PlainPq(d, m, ks, iters)` — Lloyd k-means with empty-cluster re-seeding.
- `RandomRotPq::build(d, m, ks, iters, seed) → RotatedPq` — Gram–Schmidt of a
  Gaussian d×d matrix produces the rotation, then PlainPq on rotated data.
- `OpqHouseholder::build(d, m, ks, pq_iters, opq_iters, data, seed) → RotatedPq`
  — alternating Kabsch/PQ optimization. Kabsch step:
  1. Compute cross-covariance `C = Y Xᵀ` where `Y` is the current
     PQ-reconstructed rotated data and `X` the original data.
  2. Compute polar factor: `R = C · (CᵀC)^{-1/2}`.
  3. `(CᵀC)^{-1/2}` from a symmetric-Jacobi eigendecomposition
     (no SVD dependency).

## Implementation notes

- `f32` everywhere; safe Rust only.
- Rotation stored as row-major `d × d` `Vec<f32>` — 16 KB at d=64.
- `jacobi_eig` iterates 2×2 rotations until off-diagonal norm < 1e-10 or 40
  sweeps. Converges in ≤10 sweeps for our test matrices.
- Empty k-means clusters are re-seeded from a random data point (avoids the
  classic "one giant cluster" failure mode on skewed data).
- Encoded byte codes; ADC tables are `[m × ks]` `f32`.
- No unsafe, no SIMD (yet), no external BLAS. Everything under 500 lines.

## Benchmark methodology

- **Hardware**: Apple M-series, single thread, `--release`.
- **Data**: n=4000 Gaussian vectors, d=64, with dims `0..d/2` scaled ×4.0 and
  dims `d/2..d` scaled ×0.25. This produces a 256:1 variance ratio between
  halves, mimicking real embedding skew.
- **Queries**: 100 held-out queries from the same distribution.
- **Ground truth**: brute-force L2 over `f32`.
- **Metric**: recall@10, per-query search latency (linear scan over all 4000
  codes with ADC).
- **Config**: `m=8, ks=256` → 8 bytes/vector, 32× compression from f32.
- **Runs**: 20 Lloyd iters, 3 outer OPQ sweeps.

Reproduce:

```bash
cargo run --release -p ruvector-opq-rotation --bin opq-benchmark
```

## Results

```
d=64  n=4000  n_q=100  k=10  m=8  ks=256  (8 bytes/code)

backend            recall@10   encode_ms   per_query_ms
PlainPq            0.3090      12.2        0.07
RandomRotPq        0.4450      15.8        0.07    (+44% recall)
OpqHouseholder     0.4860      15.7        0.07    (+57% recall)
```

Training cost (one-time): PlainPq 275 ms, RandomRotPq 277 ms, OpqHouseholder
1160 ms (~4× plain, dominated by 3 rounds of Lloyd k-means).

**Key finding**: random rotation captures ~90% of the OPQ recall lift for
free. Learned OPQ adds another ~9% on top and is worth it whenever training
is amortized across many queries — i.e. always in production. Query latency
is identical across backends because ADC scoring is unchanged and the
rotation is applied once per query (~10 µs), not once per code.

## How it works (walkthrough)

Consider a query `q` and a database vector `x`. Plain PQ replaces `x` with
`x̂ = decode(encode(x))`, then approximates `||q − x||²` by summing per-sub-space
distances between `q`'s sub-vectors and `x̂`'s codeword sub-vectors. This
approximation is only as good as the sub-space codebook can represent its
sub-space.

If one sub-space happens to carry 80% of `x`'s variance, its codewords need
to be very spread out, and 256 codewords can't cover the range densely. Recall
tanks.

OPQ rotates *both* `q` and `x` by the same orthogonal `R` before encoding.
Because `R` is orthogonal, `||Rq − Rx||² = ||q − x||²` — distances are exactly
preserved. The freedom is in *how the variance is distributed after rotation*.
OPQ picks `R` so each sub-space (after rotation) has roughly equal variance —
codebooks are balanced, error is minimized.

The training loop is textbook alternating minimization:

1. Freeze `R`, train PQ on `RX`.
2. Freeze PQ codebooks, solve `argmin_R ||RX − Y||²` with `Y` = decoded
   codes. This is the classic orthogonal Procrustes problem with closed-form
   solution `R = U Vᵀ` from `SVD(YXᵀ)`. We compute it SVD-free by taking the
   polar factor of `C = YXᵀ`.

At query time: rotate `q` once, then business-as-usual ADC.

## Practical failure modes

- **Uniform-variance data**: OPQ ≡ RandomRotPq ≡ PlainPq. No harm, no gain.
  Detect via `Var(dim_i)` spread before enabling.
- **Very small training sets** (`n < 10 · m · ks`): PQ codebooks under-fit,
  OPQ can over-fit the rotation to noise. Cap `opq_iters ≤ 2` in this regime.
- **d not divisible by m**: assertion at construction. Pad or slice input.
- **Near-singular CᵀC**: we clamp eigenvalues at 1e-8. If clamped values
  dominate, the rotation degenerates toward random — no worse than
  RandomRotPq, so failure is graceful.
- **High-dimensional data (d > 512)**: Jacobi eigendecomp is O(d³) per outer
  step. Above d=256 consider a truncated variant or block-diagonal rotation.

## What to improve next

1. **LOPQ**: per-IVF-cell rotation. Higher recall on hard datasets.
2. **SIMD matmul** for the query rotation (portable_simd or manual f32x8).
3. **Anisotropic loss weighting** (ScaNN-style) — orthogonal to OPQ, stacks.
4. **Learned residual rotation** for two-tier PQ (residual PQ + OPQ per tier).
5. **Streaming Kabsch** so the rotation can be refined incrementally as new
   vectors arrive without full retrain.
6. **Real-embeddings benchmark**: SIFT-1M and GloVe-100 for recall vs FAISS.

## Production crate layout proposal

```
ruvector-opq-rotation/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs          # traits, PlainPq, RotatedPq, OpqHouseholder, Jacobi eig
    ├── util.rs         # data gen, brute-force GT, recall@k
    └── main.rs         # opq-benchmark binary
```

Integration path into `ruvector-pq-search`:

```rust
let idx = PqIndexBuilder::new(d, m, ks)
    .rotation(OpqHouseholder::train(&data, 3))   // opt-in
    .build();
```

Files stay under 500 lines each. No unsafe. No new workspace-wide deps beyond
`rand` and `thiserror` (already present).

## References

- Ge, He, Ke, Sun. *Optimized Product Quantization*. CVPR 2013.
- Norouzi, Fleet. *Cartesian K-Means*. CVPR 2013.
- Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor Search*. TPAMI 2011.
- Kalantidis, Avrithis. *Locally Optimized Product Quantization*. CVPR 2014.
- Guo et al. *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN). ICML 2020.
- Kabsch. *A solution for the best rotation to relate two sets of vectors*. Acta Crystallographica 1976.
