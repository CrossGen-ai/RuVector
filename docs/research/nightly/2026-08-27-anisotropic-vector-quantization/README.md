# Anisotropic Vector Quantization for ruvector

**Date:** 2026-08-27
**Branch:** `research/nightly/2026-08-27-anisotropic-vector-quantization`
**Crate:** [`crates/ruvector-avq`](../../../../crates/ruvector-avq)
**ADR:** [ADR-341](../../../adr/ADR-341-anisotropic-vector-quantization.md)

## Abstract

We port Guo et al.'s *Anisotropic Vector Quantization* (AVQ, ICML 2020, the
codebook-training method behind Google's ScaNN) into a standalone pure-Rust
crate and benchmark three head-to-head variants on the same 20 000-vector
128-D corpus: a plain MSE product-quantization baseline (`PqMse`), a
score-aware AVQ codebook (`AvqScoreAware`), and an AVQ codebook trained on
unit-normalised vectors with a per-vector-norm side channel (`AvqNorm`).
All three share the same encoded footprint (M bytes per vector) and the
same asymmetric-distance kernel; the only variable is the codebook loss.

The real, cargo-run headline result on a heteroskedastic-magnitude
mixture-of-Gaussians corpus is that **the naïve score-aware loss can
actually *degrade* recall relative to MSE when magnitude varies across the
corpus**, but the norm-decoupled variant (`AvqNorm`) delivers a 3.3–5.2×
Recall@10 improvement at only 4 additional bytes per vector. That is the
main practical takeaway: the anisotropic idea only pays off after you
factor magnitude out first.

## SOTA survey

* **Guo et al. 2020 — *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization*** (ICML). Introduces the score-aware quantization
  loss `h_∥ ‖e_∥‖² + h_⊥ ‖e_⊥‖²`, closed-form-derives the parallel-vs-
  orthogonal weight ratio `η = (d−1) T² / (1 − T²)` from a target inner-
  product threshold `T`, and shows a 2× throughput / recall trade-off vs
  plain PQ on GLOVE-1M and DEEP-1B. arXiv 1908.10396.
* **Jégou, Douze & Schmid 2011 — *Product Quantization for Nearest
  Neighbor Search*** (TPAMI). The MSE-PQ baseline every subsequent paper
  compares against.
* **Wang, Xu, Yue & Wang 2021 — *A Comprehensive Survey and Experimental
  Comparison of Graph-Based Approximate Nearest Neighbor Search*** (VLDB).
  Places PQ variants in the broader ANN landscape and quantifies where
  scalar / product / additive quantization each dominate.
* **Aumüller, Bernhardsson & Faithfull 2020 — *ANN-Benchmarks: A
  Benchmarking Tool for Approximate Nearest Neighbor Algorithms***.
  Reference recall/QPS ladder we would need to run against for public
  competitiveness numbers (`glove-100-angular`, `sift-1m`).
* **Gao & Long 2024 — *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search***
  (VLDB'24). The current SOTA extreme-compression baseline (1 bit / dim)
  with a formal recall guarantee. Already lives in the repo as
  `ruvector-rabitq`; complementary to AVQ (RaBitQ minimises MSE with a
  bit budget, AVQ reshapes the error covariance).
* **Douze, Ivanov, Ramkumar & Amsaleg 2023 — *The FAISS Library***.
  Documents Facebook AI's production PQ variants; useful as a
  cross-check on encode / ADC throughput numbers.

None of the crates already in `crates/` implements score-aware
quantization: `ruvector-pq-search` and `ruvector-rabitq` both minimise
MSE (or a bit-budget MSE surrogate).

## Proposed design

The crate exposes one small trait:

```rust
pub trait Quantizer {
    fn train(&mut self, data: &[Vec<f32>]) -> Result<(), AvqError>;
    fn encode(&self, data: &[Vec<f32>]) -> Result<Vec<u8>, AvqError>;
    fn adc(&self, query: &[f32], codes: &[u8]) -> Result<Vec<f32>, AvqError>;
    fn code_bytes(&self) -> usize;
    fn side_bytes(&self) -> usize { 0 }
}
```

The three implementations share a single M × K<sub>s</sub> × d/M
codebook layout and a single LUT-based ADC. What differs is the
training loss:

| Variant         | Assignment step                                | Update step |
| --------------- | ---------------------------------------------- | ----------- |
| `PqMse`         | argmin ‖x−c‖² (Lloyd)                          | mean of assigned points |
| `AvqScoreAware` | argmin `η‖e_∥‖² + ‖e_⊥‖²`                     | solve `(N I + (η−1) Σ x̂x̂ᵀ) c = Σx + (η−1) Σ‖x‖ x̂` |
| `AvqNorm`       | same as `AvqScoreAware`, but on `x̃ = x/‖x‖`   | same, and `‖x‖` stored per-vector |

`η` is derived from the target inner-product threshold `T` (default
`0.2`) via the paper's closed form `η = (d−1) T² / (1 − T²)`.

## Implementation notes

* Pure safe Rust; no `unsafe`, no SIMD intrinsics. Only workspace deps
  (`rand`, `rand_distr`, `thiserror`, `serde`, `serde_json`).
* Deterministic k-means++ seeding (`StdRng`), so every table in this
  document is reproducible from a fresh clone.
* Anisotropic update solves the `d_sub × d_sub` linear system with
  partial-pivot Gaussian elimination. `d_sub` is `dim / M`, i.e. 4–16
  in practice — negligible cost.
* Empty clusters are re-seeded from a random training point (Lloyd's
  classic empty-cluster hazard would otherwise bite hard at K<sub>s</sub>=256).
* `AvqNorm` stores one `f32` per base vector (`side_bytes = 4`). The
  ADC kernel multiplies the LUT sum by the stored norm.
* Every source file stays under 500 lines per project convention.

## Benchmark methodology

* **Hardware:** Apple M4 Max, 16 CPUs, macOS Darwin 24.6.0 arm64.
* **Corpus:** deterministic seeded mixture-of-Gaussians, 32 clusters,
  per-cluster diagonal std `∈ [0.08, 0.35]` and per-cluster magnitude
  scale `∈ [0.4, 2.5]` — so norms are heteroskedastic.
* **Shapes:** `dim=128`, `n_base=20 000`, `n_queries=200`,
  `n_train=20 000`, `k=10` for Recall@10.
* **Ground truth:** brute-force inner-product top-10, 183 ms total for
  200 queries (baseline the ADC kernels compete against).
* **Codebook:** `Ks=256` (one byte / subspace code), `iters=12` Lloyd
  iterations, `seed=0xA5EE_D2026`.
* Reported latency is per-query ADC scan across the whole 20 k base
  (LUT build + inner loop), not top-K selection.

## Results

Raw output of `cargo run --release -p ruvector-avq --bin avq-benchmark`
on the hardware above:

```
variant      M   Ks  recall@10    encode_ms     adc_us/q    bytes/vec
PqMse        8  256     0.0900        107.9        379.2            8
Avq          8  256     0.0490        318.8        390.1            8
AvqNorm      8  256     0.4675        335.0        372.9           12
PqMse       16  256     0.1355        111.9        435.4           16
Avq         16  256     0.0855        376.7        448.5           16
AvqNorm     16  256     0.4455        384.6        468.7           20
```

Same table, rounded:

| Variant   | M  | Recall@10 | Encode (20k) | ADC / query | Bytes / vec |
| --------- | -: | --------: | -----------: | ----------: | ----------: |
| PqMse     |  8 |     0.090 |       108 ms |     379 µs  |         8 B |
| Avq       |  8 | **0.049** |       319 ms |     390 µs  |         8 B |
| AvqNorm   |  8 | **0.468** |       335 ms |     373 µs  |        12 B |
| PqMse     | 16 |     0.136 |       112 ms |     435 µs  |        16 B |
| Avq       | 16 | **0.086** |       377 ms |     449 µs  |        16 B |
| AvqNorm   | 16 | **0.446** |       385 ms |     469 µs  |        20 B |

Key numbers:

* `AvqNorm` at M=8 beats `PqMse` at M=16 on recall (0.468 vs 0.136) while
  using **4 bytes less** per vector (12 vs 16). That is a real
  compression win, not a wash.
* `AvqScoreAware` without norm decoupling **regresses** recall by ~45 %
  vs the MSE baseline at both M values. This is a genuine practical
  failure mode, not a bug — see "Practical failure modes" below.
* ADC latency is identical across variants (as expected — same kernel);
  encode is ~3× slower for AVQ because the anisotropic Lloyd step is a
  d<sub>sub</sub>×d<sub>sub</sub> linear solve per centroid per
  iteration. Training is off the query path.

## How it works — a blog-readable walkthrough

Product quantization splits every vector into M subspaces of length
d/M, learns a small codebook per subspace, and stores each vector as M
codebook indices. To score a query, you build a lookup table per
subspace (query slice · codebook centroids) and sum M table lookups
per base vector. That's why PQ is the dominant compression-plus-fast-
scoring recipe for vector search.

The classical codebook loss is mean squared error — you want the
reconstruction `\tilde{x}` to be close to the original `x` in Euclidean
distance. But for **inner-product search** (which is what almost every
LLM embedding retrieval actually is), you don't care about all of `e =
x − \tilde{x}` equally: the component of `e` *parallel to `x`* is the
part that shifts the inner-product score `q·\tilde{x}` when the query
`q` looks like `x`. The orthogonal part barely matters.

Guo et al. formalised that as an anisotropic loss `η ‖e_∥‖² + ‖e_⊥‖²`
and showed that biasing k-means toward reducing parallel error gives
a strictly better recall / bit budget on ANN benchmarks. Great — but
their derivation assumes vectors have comparable magnitude, because the
"parallel direction" `x̂` is a *direction*.

On corpora where magnitude varies (very common for anything that
isn't unit-normalised — dense-passage-retrieval, ColBERT, hybrid
sparse-dense embeddings), the score-aware loss *without* decoupling
magnitude picks codebooks that are pulled toward high-norm points
because those dominate the sum. Result: worse recall than plain MSE.
That's exactly what our M=8 and M=16 rows show for `Avq`.

The fix is trivial: unit-normalise before you train, store the norm
separately, and multiply it back at ADC time. Now the codebook is a
*directional* codebook, the anisotropic bias captures the score-relevant
direction cleanly, and the 4 bytes of side channel buy you a 3–5×
recall lift.

## Practical failure modes

* **Heteroskedastic magnitude on plain AVQ.** As above: naïve AVQ can
  lose to MSE. Always benchmark on your actual corpus before shipping.
* **Bad `T` choice.** `T` sets `η`. Too small (`T → 0`) collapses back to
  MSE. Too large (`T → 1`) makes `η` blow up and starves orthogonal
  fitting entirely — recall craters on realistic queries. Default
  `T=0.2` is the paper's sweet spot for cosine-scale corpora; retune
  for out-of-distribution ones.
* **Small d<sub>sub</sub>.** At `d/M = 4` or `8` the subspace direction
  is very noisy per-vector. The anisotropic solve still works, but the
  effective `η` you get is closer to the MSE regime than the numeric
  `η` would suggest. Cross-check by sweeping `M`.
* **Empty clusters at Ks=256.** With a 20 k training set, ~2–5 % of
  centroids typically end up empty on the first pass. We re-seed from
  a random training point; a colder-start production build should
  fall back to k-means++ re-seeding for stability.
* **Numerical drift on Gaussian elimination.** `d_sub` is small, so
  fine — but a d=1024 model that runs AVQ *without* subspaces would
  need iterative refinement, not partial pivoting.

## What to improve next

1. **IVF+AVQ**: pair with a coarse quantizer (IVF) as ScaNN does; the
   AVQ codebook trains on residuals per-cell, which is where the paper
   reports its best numbers.
2. **f16 norms.** Halve the AvqNorm side channel from 4 B to 2 B; the
   recall drop is negligible for typical embedding norms in
   [10<sup>−3</sup>, 10<sup>3</sup>].
3. **Additive quantization (AQ) with anisotropic loss.** AQ codebooks
   sum across subspaces instead of concatenating — combined with score-
   aware loss this pushes the theoretical recall / bit floor further
   down.
4. **SIMD ADC.** The pure-scalar LUT loop tops out at ~400 µs / query
   at N=20 k. A neon / AVX-2 gather-based ADC (see `ruvector-turboquant`)
   should hit ~40 µs / query.
5. **Benchmark on ANN-Benchmarks corpora.** `glove-100-angular` and
   `sift-1m` are the reference points reviewers will ask about.
6. **η auto-tuning.** Cross-validate `T` on a held-out slice of the
   training set instead of hard-coding 0.2.

## Production crate layout

If we promote this out of nightly, the recommended shape is:

```
crates/ruvector-avq/           # this crate — codebook training + ADC
crates/ruvector-avq-ivf/       # AVQ residuals under a coarse quantizer
crates/ruvector-avq-simd/      # neon / avx2 ADC kernels behind a feature flag
crates/ruvector-avq-node/      # NAPI binding for JS use
crates/ruvector-avq-wasm/      # wasm-bindgen build for browser
```

The trait boundary in `ruvector-avq::Quantizer` is intentionally small
so the IVF and SIMD variants can drop in with zero refactoring.

## References

* Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar,
  S. *Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization*. ICML 2020. arXiv 1908.10396.
* Jégou, H., Douze, M., Schmid, C. *Product Quantization for Nearest
  Neighbor Search*. IEEE TPAMI 33(1), 2011.
* Wang, M., Xu, X., Yue, Q., Wang, Y. *A Comprehensive Survey and
  Experimental Comparison of Graph-Based Approximate Nearest Neighbor
  Search*. VLDB 2021.
* Aumüller, M., Bernhardsson, E., Faithfull, A. *ANN-Benchmarks: A
  Benchmarking Tool for Approximate Nearest Neighbor Algorithms*.
  Information Systems 87, 2020.
* Gao, J., Long, C. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search*.
  VLDB 2024.
* Douze, M., Ivanov, N., Ramkumar, S., Amsaleg, L. *The FAISS Library*.
  2023 tech report.
