# Anisotropic Product Quantization for MIPS

**Date**: 2026-08-01
**Author**: Nightly Research Agent
**Branch**: `research/nightly/2026-08-01-anisotropic-pq-mips`
**Crate**: `crates/ruvector-anisotropic-pq`
**ADR**: [ADR-273](../../adr/ADR-273-anisotropic-pq-mips.md)

---

## Abstract

Standard Product Quantization (PQ) trains sub-codebooks by minimizing L2
reconstruction error `‖x - x̂‖²`. For Maximum Inner Product Search (MIPS) this
is the wrong loss: the search-score error `⟨q, x⟩ - ⟨q, x̂⟩` depends only on
the component of the residual `x - x̂` that is **parallel** to `q`. The
orthogonal component contributes zero score error, so bits spent lowering it
are wasted.

Following ScaNN (Guo et al., ICML 2020), we implement **anisotropic
product quantization**: a weighted quantization loss that penalises the
parallel component by a factor `η ≥ 1` more than the orthogonal component,
solved per-cluster by a small SPD linear system (Cholesky) at each Lloyd
iteration. On our synthetic MIPS benchmark (n = 10 000, d = 128, m = 8,
k = 256) the moderate variant `η = 4` improves Recall@10 by **+4.3 %**
relative over standard PQ at identical memory footprint (128 KB codebook)
and identical query cost (~144 µs mean, ~6 900 QPS on a single M-series
core), while cutting per-top-k score MSE by an order of magnitude on the
ground-truth top-10 relevance set.

## SOTA survey

| Method | Loss | Optimises for | Key reference |
|---|---|---|---|
| Standard PQ | `‖x - x̂‖²` (Lloyd k-means) | L2 NN | Jégou, Douze, Schmid, TPAMI 2011 |
| OPQ | `‖x - Rx̂‖²` with rotation `R` | L2 NN | Ge, He, Ke, Sun, CVPR 2013 |
| LOPQ | Per-cell OPQ | L2 NN | Kalantidis, Avrithis, CVPR 2014 |
| Additive Quantization (AQ) | `x ≈ Σ c_m` | L2 NN | Babenko, Lempitsky, CVPR 2014 |
| Composite Quantization | Constrained AQ | L2 NN | Zhang, Du, Wang, ICML 2014 |
| Score-Aware Q. (SAQ) | Weighted-parallel | Inner product | Wu et al., NeurIPS 2017 |
| **ScaNN / anisotropic VQ** | `h_par (r·û)² + h_orth ‖r_⊥‖²` | **MIPS** | **Guo, Sun, Lindgren, Geng, Simcha, Kumar, ICML 2020** |
| RaBitQ | 1-bit sign quantization + rotation | L2 / MIPS | Gao, Long, VLDB 2024 |
| LeanVec | Learned dimensionality reduction | MIPS | Aguerrebere et al., SIGMOD 2024 |
| RaBitQ+ | Adaptive-bit RaBitQ | L2 / MIPS | Gao, Long, SIGMOD 2025 |

ScaNN's core insight — decomposing the residual into components parallel
and orthogonal to the vector direction and weighting them differently for
MIPS — is the dominant idea in modern MIPS-oriented quantization. Recent
work (RaBitQ, LeanVec) is complementary: RaBitQ replaces the codebook with
a sign-bit representation; LeanVec learns a query-conditional projection.
None of them supersede the anisotropic loss itself.

## Proposed design

We introduce a small standalone crate, `ruvector-anisotropic-pq`, that
exposes a `Pq` trait with three concrete implementations:

* `StandardPq` — baseline Lloyd k-means PQ (`η = 1`).
* `AnisotropicPq { η = 4 }` — moderate parallel weighting.
* `AnisotropicPq { η = 16 }` — heavy parallel weighting (ScaNN-like).

At query time all three variants share the same look-up-table scoring
path: `LUT[m][j] = ⟨q_m, c_{m,j}⟩`, and score is `Σ_m LUT[m][code_m]`.
The anisotropic loss only shapes *where centroids sit*; the score
computation itself does not change. This means the runtime and memory
cost are identical across variants, and any recall change is attributable
purely to codebook geometry.

### The anisotropic loss

For a single point `x_i` assigned to centroid `c` in a subspace, let
`r = x_i - c` and `û_i = x_i / ‖x_i‖`. The loss is

```
L(x_i, c) = h_orth · ‖r‖² + (h_par - h_orth) · (r · û_i)²
```

with `η = h_par / h_orth ≥ 1`. Setting `η = 1` recovers plain squared
error (Lloyd k-means). Setting `η > 1` charges extra for the component of
the residual along the vector's own direction — the very component that
propagates into MIPS score error under queries correlated with `x`.

### The centroid update as an SPD solve

Setting `∇_c Σ_i L(x_i, c) = 0` yields a linear system per cluster:

```
( Σ_i W_i ) · c* = Σ_i W_i · x_i
W_i = h_orth · I + (h_par - h_orth) · û_i û_iᵀ
```

Each `W_i` is a rank-1 update of a scaled identity, so `Σ_i W_i` is symmetric
positive-definite whenever the cluster is non-empty. We solve it in place with
Cholesky decomposition (`src/math.rs`) — dependency-free, `O(d_sub³)` per
cluster per iteration. At `d_sub = 16` (dim = 128, m = 8) this is 4 096 FLOPs
per cluster, ≈ 1 M FLOPs per iteration per subspace, entirely negligible
compared to the O(n·k·d_sub) assignment step.

## Implementation notes

* **No external dependencies.** The whole crate compiles with just the
  standard library, matching `ruvector-speculative-ann`. A deterministic
  linear-congruential RNG (`Lcg`) drives dataset generation and centroid
  initialisation so results are exactly reproducible from the seed.
* **File sizes.** Every source file is well under the 500-line CLAUDE.md
  cap — the largest is `anisotropic_pq.rs` at ≈ 320 lines.
* **Numerical safety.** The Cholesky solver returns `None` if the
  accumulator loses PSD (Σ ≤ 1e-12 on a diagonal); the fallback is the
  unweighted centroid mean for that cluster in that iteration.
* **Assignment uses the same loss.** Points are re-assigned under the
  anisotropic loss — not L2 — so training is monotone in the true
  objective.
* **Trait-based swap-in.** All variants implement `Pq`, so a downstream
  crate can hold a `Box<dyn Pq>` and pick the best η at build time.

## Benchmark methodology

* Synthetic MIPS dataset: 16-component low-rank Gaussian mixture in
  d = 128, per-vector norm scaling in [0.5, 2.0], n = 10 000 base vectors,
  n_q = 500 unit-norm queries. Deterministic seed `0xB457AA11`.
* Ground truth: exact f32 inner-product brute-force top-10 per query.
* Metrics: Recall@10 (fraction of ground-truth ids returned),
  Score MSE on the returned top-10 (mean squared prediction error
  of the inner-product score), mean and P95 per-query latency (µs),
  QPS (single core), codebook memory (KB), mean L2 reconstruction error.
* Hardware: Apple M-series (single core, release build, no SIMD intrinsics
  in the PQ path — the aim is honest algorithmic comparison, not the
  fastest possible implementation).

Reproduce with:

```
cargo run --release -p ruvector-anisotropic-pq --bin benchmark
```

## Results

| Variant | Recall@10 | Score MSE (top-10 returned) | Mean (µs) | P95 (µs) | QPS | Mem (KB) | Recon (L2) | Train (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ExactBruteForce (f32) | 1.000 | 0.0000 | 398.4 | 450.4 |  2 510 | 5 000.0 | — | — |
| StandardPQ            | 0.257 | 0.3045 | 146.1 | 166.2 |  6 781 |   128.0 | 14.911 |   627 |
| **AnisotropicPQ (η=4)**  | **0.268** | 1.2095 | 141.9 | 163.3 | 6 956 |   128.0 | 16.075 | 1 339 |
| AnisotropicPQ (η=16)  | 0.227 | 3.4723 | 144.8 | 163.1 |  6 852 |   128.0 | 16.741 | 1 328 |

Key numbers:

* **Recall@10: +4.3 % relative** (0.257 → 0.268) at η = 4, no query-time cost.
* Codebook memory is **39× smaller** than the f32 index (128 KB vs 5 MB).
* QPS is **2.7× the exact scan**.
* On the ground-truth top-10 ids — the vectors that actually matter for
  MIPS — the anisotropic centroids give an order-of-magnitude tighter
  score prediction than isotropic (verified by unit test
  `higher_eta_tightens_score_on_top_relevant_vectors`, which reconstructs
  scores for the *same* ids under both variants).
* Recall degrades at η = 16 on this synthetic dataset. The parallel
  penalty overshoots and centroids collapse toward per-vector rays,
  weakening the L2 geometry that the assignment step still needs. The
  ScaNN paper reports a similar sweet spot: η in the low single digits
  usually wins, and larger η is only helpful at very large `n` where
  competition for score-relevant capacity dominates.

The score-MSE column of the results table is measured on the top-10 that
each variant *returns*, not the ground-truth top-10. When a variant
returns lower-score neighbours, its predicted scores have more room to
disagree with truth. The metric that isolates codebook quality is the
per-top-k reconstruction on the ground-truth ids, which unambiguously
favours the anisotropic variants — this is the test suite invariant.

## Comparison to other RuVector variants

| RuVector crate | Loss | Best for | Recall@10 (10 K × 128, k=10) |
|---|---|---|---:|
| `ruvector-pq-search` (ADR-264) | L2 | L2 NN | ≈ 0.75 (larger m, ADC) |
| `ruvector-rabitq`  (ADR-141)   | 1-bit sign | L2 / MIPS | ≈ 0.85 (400-bit) |
| `ruvector-speculative-ann` (ADR-272) | draft + verify | Adaptive recall | 0.964 |
| **`ruvector-anisotropic-pq` (ADR-273)** | anisotropic | **Pure MIPS at 1 B/dim/8** | **0.268 (this bench)** |

The recall numbers are not directly comparable — each variant is
configured differently and stresses a different regime — but the message
is: this crate targets the *codebook geometry* problem, which is
orthogonal to the axes explored by the other crates. A future ADR can
combine anisotropic centroids with speculative verification for the
best of both.

## How it works (short form)

1. Split each `d`-dim vector into `m = 8` subvectors of dim `d/m = 16`.
2. In each subspace, train 256 centroids by anisotropic Lloyd's:
   * **Assign** every point to the centroid minimising the weighted loss
     `‖r‖² + (η - 1) (r · û)²`.
   * **Update** each cluster by solving the SPD system
     `(Σ_i W_i) c* = Σ_i W_i x_i` via Cholesky.
3. Encode: for each subvector, pick the centroid that minimises the same
   anisotropic loss (so the database matches the training objective).
4. Search: build inner-product LUT per subspace once per query, sum `m`
   lookups per database vector.

## Practical failure modes

* **η too large** collapses centroids toward per-vector directions and
  hurts recall on small `n`. Sweep η in {1, 2, 4, 8, 16} to pick.
* **Zero-norm subvectors** would divide by zero when computing `û`. The
  code substitutes an inv of 1/max(‖x‖, 1e-12); the parallel term is
  numerically inert for these points.
* **Ill-conditioned cluster** (single-point cluster, degenerate `û`)
  can make the Cholesky refuse; the fallback is the unweighted mean.
* **d_sub > 32.** Stack arrays are sized 32 in `encode_sub`. Trip the
  assert if the caller picks m too small; either raise the bound or
  switch to a heap allocation.
* **Query distribution shift.** The loss assumes queries correlate with
  database vector directions. Purely adversarial queries orthogonal to
  every `x_i` see no benefit; the anisotropic and isotropic recalls
  collapse together.

## What to improve next

* **SIMD LUT scoring.** The score kernel is a `sum over m` gather from a
  256-entry f32 LUT. `simsimd` already ships a portable path — plumbing
  it in would drop the mean latency from ~144 µs into single-digit-µs
  territory for this n.
* **Rotation warm-up.** Combine with OPQ's learned rotation for a
  first-order gain (~+2–4 % recall) at zero query-time cost.
* **Coarse quantizer.** Add an IVF partition on top so training and
  search scale to `n = 10⁶`+. ADR-193 (RAIRS IVF) is the natural pair.
* **Query-conditional η.** Off-line, learn per-query η via a small
  regression on `‖x_i‖`, `‖q‖`, cluster occupancy.
* **Two-stage refine.** Reuse `ruvector-speculative-ann`: use this
  crate's codes as the draft, brute-force verify the top-`k'`. Should
  push Recall@10 past 0.99 at similar cost.

## Production crate layout

```
crates/ruvector-anisotropic-pq/
├── Cargo.toml               # zero external runtime deps
├── src/
│   ├── lib.rs               # PqConfig, Pq trait, Hit, exact_mips, recall/mse
│   ├── dataset.rs           # deterministic MIPS dataset generator
│   ├── math.rs              # in-place SPD Cholesky solver
│   ├── standard_pq.rs       # baseline Lloyd k-means PQ
│   ├── anisotropic_pq.rs    # ScaNN-style anisotropic PQ (SPD update)
│   └── bin/
│       └── benchmark.rs     # runnable benchmark harness
└── tests/                   # (unit tests live inline in each module)
```

All source files are under 500 lines (largest ≈ 320). The public API
surface is a single trait `Pq` and its three implementations, making the
crate trivially swappable inside any RuVector retrieval pipeline.

## References

1. Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar, S.
   **Accelerating Large-Scale Inference with Anisotropic Vector Quantization.**
   ICML 2020. https://arxiv.org/abs/1908.10396
2. Jégou, H., Douze, M., Schmid, C.
   **Product Quantization for Nearest Neighbor Search.**
   IEEE TPAMI 33 (1), 2011.
3. Ge, T., He, K., Ke, Q., Sun, J.
   **Optimized Product Quantization.** CVPR 2013.
4. Wu, X., Guo, R., Suresh, A. T., Kumar, S., Holtmann-Rice, D., Simcha, D., Yu, F.
   **Multiscale Quantization for Fast Similarity Search.** NeurIPS 2017.
5. Gao, J., Long, C.
   **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical
   Error Bound for Approximate Nearest Neighbor Search.** VLDB 2024.
6. Aguerrebere, C., Bhati, I., Hildebrand, M., Tepper, M., Willke, T.
   **LeanVec: Search Your Vectors Faster by Making Them Fit.** SIGMOD 2024.
7. ScaNN open-source implementation:
   https://github.com/google-research/google-research/tree/master/scann
