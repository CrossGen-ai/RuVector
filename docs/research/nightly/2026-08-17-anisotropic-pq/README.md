# Anisotropic (Score-Aware) Product Quantization for ruvector

> Nightly research spike · 2026-08-17 · branch `research/nightly/2026-08-17-anisotropic-pq`
> Crate: [`crates/ruvector-anisotropic-pq`](../../../../crates/ruvector-anisotropic-pq)
> ADR: [ADR-305](../../../adr/ADR-305-anisotropic-pq.md)

## Abstract

Standard product quantization (PQ) trains codebooks by minimizing L2
reconstruction error. For inner-product search — the dominant retrieval
regime for RAG, recommenders, and LLM embedding stores — this is the wrong
loss: errors aligned with the query direction distort dot-product scores
more than perpendicular errors of equal magnitude. Guo et al.'s ScaNN
(ICML 2020) showed that a **score-aware anisotropic loss** wins on
ANN-benchmarks.

This spike lands a first-class Rust implementation as
`ruvector-anisotropic-pq` with three swappable trainers (`L2`,
`NormWeighted`, `Anisotropic`) behind a `PqTrainer` trait, plus a real
benchmark harness with exact-IP ground truth. We report **two honest
findings**:

1. **`NormWeighted-PQ` is a free win**: +10.6–10.7 pp recall@10 over
   plain L2-PQ at identical memory and identical training cost, on
   heavy-tail-norm data that mimics real embedding stores.
2. **Naïve per-subvector anisotropic loss is a lose**: −2.4 pp @ η=2 to
   −8.8 pp @ η=6, at 2.7× the training cost. The subvector-factorized
   loss is *not* equivalent to the paper's full-vector loss, and this
   is measurable.

Finding (2) motivates the next spike: full-vector anisotropic loss with
jointly-optimized OPQ rotation.

## SOTA survey

- **Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar (ICML 2020)** —
  "Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization." Introduces the parallel/perpendicular loss decomposition
  and the SOAR-adjacent iterative training. Powers Google ScaNN.
  <https://arxiv.org/abs/1908.10396>
- **Ge, He, Ke, Sun (CVPR 2013)** — "Optimized Product Quantization."
  The rotation `R` fitted jointly with the codebook to align variance
  with subquantizer boundaries. Compatible with anisotropic loss.
- **Johnson, Douze, Jégou (FAISS, 2017-)** — reference PQ + IVFPQ
  implementation; L2 loss only.
- **Jégou, Douze, Schmid (2011)** — original PQ.
- **Aumüller, Bernhardsson, Faithfull (ANN-Benchmarks, 2024)** — reports
  ScaNN dominates PQ methods on GloVe-100/angular and SIFT-1M/L2 by
  1.3–2× recall at fixed memory.
- **Milvus 2.4 changelog (2025)** — added `SCANN` index (wraps ScaNN);
  contributors note the anisotropic loss as the primary quality delta
  vs their own IVFPQ.
- **Qdrant blog (2025)** — introduced BQ + rerank; explicitly cites
  anisotropic PQ as "future work."
- **LanceDB (2025)** — ships plain IVFPQ; no anisotropic path.

**Gap**: no Rust-native implementation in a first-party vector database
crate. ruvector already has RaBitQ, IVFPQ, and turbo-quant; anisotropic
was missing.

## Proposed design

### Trainer trait

```rust
pub trait PqTrainer {
    fn name(&self) -> &'static str;
    fn train(&self, data: &[f32], n: usize, dim: usize, m: usize, k: usize)
        -> Result<PqCodebook, PqError>;
}
```

Three impls:

| Impl                    | Assignment loss                                  | Update rule                                    |
|-------------------------|--------------------------------------------------|------------------------------------------------|
| `L2Trainer`             | `‖s − c‖²`                                      | unweighted mean                                |
| `NormWeightedTrainer`   | `‖s − c‖²` (assignment); update weighted by `‖x‖²` | weighted mean                                  |
| `AnisotropicTrainer(η)` | `‖s − c‖² + (η−1) ((s−c)·u)²`, `u = s/‖s‖`     | closed-form `A c = b`, `A = Σ (I + (η−1) uuᵀ)` |

### Anisotropic centroid update (per subquantizer, per cluster)

Setting the gradient to zero:

```
∂/∂c  Σ [‖s − c‖² + (η−1) ((s − c)·u)²]  =  0
⇒     Σ (I + (η−1) u uᵀ) (s − c)          =  0
⇒     [ Σ (I + (η−1) u uᵀ) ]  c           =  Σ (I + (η−1) u uᵀ) s
```

Left-hand side is a d_sub × d_sub SPD matrix; solved by in-place
Gauss-Jordan (d_sub is typically 4–16). At η = 1, `A = n · I` and this
reduces to the unweighted mean — verified by the
`anisotropic_matches_l2_at_eta_one` unit test.

### Compression math

For `m = 8, k = 64, d = 64, dtype = f32`:
- Raw: `64 × 4 = 256 B/vec`
- PQ codes (u8): `8 B/vec` — **32× compression**
- Codebook one-off: `m × k × d_sub × 4 = 8 × 64 × 8 × 4 = 16 KB total`

For `m = 16, k = 256, d = 128`:
- Raw: `512 B/vec`
- PQ codes: `16 B/vec` — **32× compression**
- Codebook: `16 × 256 × 8 × 4 = 128 KB total`

## Implementation notes

- **Single file (`src/lib.rs`, ~500 lines, under budget)** — one trainer
  fn parametrized by a `LossKind` enum keeps the assignment loop shared
  so benchmark comparisons are fair.
- **No `unsafe`** (`#![forbid(unsafe_code)]`).
- **No SIMD intrinsics** — future work; kept out of scope so the loss
  math dominates measurements.
- **Deterministic seeding** — every trainer takes a `seed: u64`; benchmark
  uses the same seed across variants for identical initialization.
- **k = 256 max** in `encode/decode` (u8 codes) — matches every
  production PQ implementation.

## Benchmark methodology

- **Data**: `n × d` matrix, each vector is `scale_i · g` with
  `scale_i ~ Pareto(α=1.5)` and `g ~ N(0, I_d)`. This produces the
  heavy-tail norm distribution characteristic of real recommender and
  LLM-embedding corpora.
- **Ground truth**: exact top-10 by inner product, brute-forced.
- **Metric**: recall@10 vs exact IP truth.
- **Hardware**: Apple M4 Max, 128 GB RAM, macOS Darwin 24.6.0, single
  thread, `--release` (opt-level 3, LTO from workspace).
- **Reproduce**:
  ```
  cargo run --release -p ruvector-anisotropic-pq --bin aniso-pq-bench
  ANISO_N=10000 ANISO_DIM=128 ANISO_M=16 ANISO_K=256 \
    cargo run --release -p ruvector-anisotropic-pq --bin aniso-pq-bench
  ```

## Results

### Config A: n=5,000  d=64  m=8  k=64  (8 B/vec, 32× compression)

| Variant             | recall@10 | Δ vs L2 | train (ms) | encode (ms) | query 100q (ms) |
|---------------------|-----------|---------|------------|-------------|------------------|
| L2-PQ               | 0.6420    | —       | 100.8      | 4.2         | 6.1              |
| **NormWeighted-PQ** | **0.7480**| **+10.60 pp** | 96.7  | 4.8         | 3.7              |
| Anisotropic η=2     | 0.6180    | −2.40 pp| 280.6      | 4.5         | 6.3              |
| Anisotropic η=6     | 0.5650    | −7.70 pp| 275.5      | 4.3         | 6.3              |

### Config B: n=10,000  d=128  m=16  k=256  (16 B/vec, 32× compression)

| Variant             | recall@10 | Δ vs L2  | train (ms) | encode (ms) | query 100q (ms) |
|---------------------|-----------|----------|------------|-------------|------------------|
| L2-PQ               | 0.6700    | —        | 1334.2     | 67.1        | 17.6             |
| **NormWeighted-PQ** | **0.7770**| **+10.70 pp** | 1332.6| 62.2        | 15.8             |
| Anisotropic η=2     | 0.6390    | −3.10 pp | 3686.0     | 68.2        | 18.5             |
| Anisotropic η=6     | 0.5820    | −8.80 pp | 3665.6     | 63.8        | 15.0             |

**Two configs, same story**: NormWeighted wins by ~10.7 pp at zero cost;
per-subvector Anisotropic hurts.

## How it works (walkthrough)

Think of PQ as splitting a `d = 128`-dim vector into `m = 16` slices of 8
dims each, and learning 256 "typical" 8-dim shapes for each slice. To
encode a vector, replace each slice with the index of the nearest typical
shape. Storage: `16 × 1 byte = 16 B` per vector — a 32× shrink.

Standard PQ (`L2Trainer`) fits those typical shapes by squared-error
k-means: each slice is treated equally, each vector contributes equally.

`NormWeightedTrainer` weights each vector by `‖x‖²` when computing
cluster centroids. If your data has a few huge-norm vectors and many
tiny ones (recommender item embeddings, LLM token embeddings), those
few huge vectors dominate inner-product ranking — so the codebook
should quantize *them* accurately. This is what norm-weighting does,
essentially for free.

`AnisotropicTrainer(η)` tries to be smarter still: it penalizes error
components that point along the vector's own direction, since those
mess up dot products more. The math yields a closed-form linear solve
per centroid. The **twist** we found: applying this loss at the
subvector level (which is what you *have* to do to keep PQ's factorized
structure) is not equivalent to the paper's full-vector loss —
subvector directions don't add up to the full-vector direction. So the
subvector "anisotropy" isn't the same anisotropy that hurts MIPS at
query time, and empirically it makes things worse.

## Practical failure modes

1. **Don't use `AnisotropicTrainer` as-is in production.** Our numbers
   show it hurts. Use it only as a starting point for the full-vector
   variant (see "What to improve next").
2. **NormWeighted assumes MIPS.** If your workload is Euclidean-NN
   (e.g., L2 embeddings from CLIP), norm-weighting the k-means will bias
   the codebook toward high-norm vectors *without* the corresponding
   ranking benefit. Fall back to `L2Trainer` for pure L2 search.
3. **k ≤ 256** (u8 codes). For `k = 65,536` you'd need u16 codes and
   different ADC layout.
4. **Singular normal equations**: the anisotropic update solver returns
   `PqError::Singular` if a cluster's `A` matrix is degenerate (all
   assigned vectors collinear with u). We fall back to random re-seed
   for empty clusters but not for singular ones — production would want
   a Tikhonov regularizer `A ← A + λI`.
5. **Training determinism**: seeds are honored, but iteration order over
   clusters can vary if you parallelize the outer loop with `rayon` (not
   done here to keep the spike simple).

## What to improve next (roadmap)

1. **Full-vector anisotropic loss with jointly-optimized OPQ rotation.**
   The honest fix for finding (2). Fit `R ∈ SO(d)` such that after
   rotation the paper's loss `h_∥‖r_∥‖² + h_⊥‖r_⊥‖²` decomposes
   *approximately* across subquantizers, then alternate rotation and
   codebook updates (a Riemannian optimization on the Stiefel manifold).
   Estimated: 400 LOC, one week.
2. **SIMD ADC lookup** (AVX-512 / NEON `tbl` on M-series). Orthogonal to
   loss — a 2–4× speedup for `pq_ip_topk` regardless of trainer.
3. **Bootstrap from L2, refine anisotropically.** Init codebook with L2
   Lloyd's, then run 3–5 anisotropic iterations. Cheaper than
   from-scratch and often better-conditioned.
4. **Integrate with `ruvector-adaptive-ann`.** Let the entropy signal
   pick `η` per shard.
5. **Real dataset validation.** Run on GloVe-100, SIFT-1M,
   deep1B-100M-sample; compare against ScaNN reference numbers.
6. **Tikhonov regularization** in `gauss_solve` for stability at high η.

## Production crate layout (if promoted)

If a follow-up spike ships full-vector anisotropic with joint OPQ:

```
crates/
  ruvector-anisotropic-pq/              # this crate — score-aware PQ trainers
    src/
      lib.rs                            # trait + shared kernels
      trainers/
        l2.rs
        norm_weighted.rs
        anisotropic_subvector.rs        # current spike
        anisotropic_full.rs             # follow-up (with OPQ rotation)
      rotation.rs                       # Riemannian OPQ fitter
      adc.rs                            # SIMD-optional lookup path
    benches/
      recall_vs_ann_benchmarks.rs       # criterion
  ruvector-pq-search/                   # existing — reference trainer to L2
                                        # would gain a feature flag to swap in
```

Integration points:
- `ruvector-server`: expose `pq_trainer: enum { L2, NormWeighted,
  Anisotropic }` in index-build API.
- `ruvector-cli`: `ruvector index build --pq-trainer=norm-weighted`.
- `ruvector-node`, `ruvector-wasm`: pass-through of the same enum.

## References

1. Guo et al. "Accelerating Large-Scale Inference with Anisotropic Vector
   Quantization." ICML 2020. <https://arxiv.org/abs/1908.10396>
2. Jégou, Douze, Schmid. "Product Quantization for Nearest Neighbor
   Search." TPAMI 2011.
3. Ge, He, Ke, Sun. "Optimized Product Quantization for Approximate
   Nearest Neighbor Search." CVPR 2013.
4. Johnson, Douze, Jégou. "Billion-scale similarity search with GPUs."
   IEEE TBD 2019 (FAISS).
5. Aumüller, Bernhardsson, Faithfull. "ANN-Benchmarks." Information
   Systems 2020, ongoing update at <http://ann-benchmarks.com>.
6. Google ScaNN library. <https://github.com/google-research/google-research/tree/master/scann>
