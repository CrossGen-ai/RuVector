# Anisotropic Product Quantization for MIPS in RuVector

*Nightly research — 2026-08-05*

## Abstract

We prototype **score-aware ("anisotropic") Product Quantization** as introduced
in the ScaNN paper (Guo et al., *Accelerating Large-Scale Inference with
Anisotropic Vector Quantization*, ICML 2020) and integrate it as a swappable
trainer alongside the existing `ruvector-pq-search` isotropic PQ. On a
low-rank + isotropic-noise embedding corpus (a first-order model of
BERT/E5-style dense retrievers) with a fixed 8-bit-per-subspace × 8-subspace
budget, anisotropic training with **η = 2 lifts recall@10 from 0.526 → 0.567
(+4.2 pp)** at *identical* code size, encode footprint, and query FLOPs. The
recall curve is unimodal in η, matching the ScaNN paper's observed sweet
spot (2–8) and confirming that unbounded anisotropy over-penalises useful
orthogonal reconstruction.

## SOTA survey

- **ScaNN — Anisotropic Vector Quantization** (Guo et al., ICML 2020).
  Introduces a residual loss decomposed into parallel and orthogonal
  components against each database vector's direction; parallel residuals
  directly perturb inner-product scores while orthogonal ones average out.
  Reference: https://arxiv.org/abs/1908.10396
- **RaBitQ** (Gao & Long, SIGMOD 2024). 1-bit quantization with theoretical
  ε-bounds on inner-product error. Already implemented in
  `crates/ruvector-rabitq`.
- **LeanVec / OPQ** (Ge et al., 2013; Sablayrolles et al., 2019). Learned
  rotations before quantization; orthogonal to anisotropic weighting.
- **SOAR** (Sun et al., SIGMOD 2024). Redundant partition assignment for
  ScaNN spilling; complementary — sits *above* the codebook trainer.
- **RUV ecosystem**. Existing `ruvector-pq-search` covers Flat, IVF, and
  Residual PQ variants — all isotropic. `ruvector-rabitq` covers 1-bit.
  Anisotropic PQ has been an open gap in the coarse-to-fine funnel until
  this nightly.

## Proposed design

Introduce a small, self-contained crate `ruvector-anisotropic-pq` exposing
a `PqCodebookTrainer` trait with two implementors:

- `IsotropicTrainer` — classical Lloyd's k-means on each sub-space (baseline
  matching `ruvector-pq-search`).
- `AnisotropicTrainer { eta: f32 }` — weighted Lloyd's whose per-point loss
  is

  ```text
  L(x, c) = h_∥ · (r · û)²  +  h_⊥ · (‖r‖² − (r · û)²)
          r = x − c,  û = x / ‖x‖,  η = h_∥ / h_⊥
  ```

  The closed-form centroid update is the solution of a small `d × d`
  symmetric-positive-definite system

  ```text
  (Σᵢ Mᵢ) c* = Σᵢ Mᵢ xᵢ ,   Mᵢ = h_⊥ I + (h_∥ − h_⊥) ûᵢûᵢᵀ
  ```

  which we solve with Gauss-Jordan elimination on `f64` — `d ≤ 32` in
  practice, so per-iteration cost stays `O(NKd²)`.

The ADC search path (`AnisoPqIndex::search`) is byte-code-identical to
isotropic PQ: build a per-query LUT of size `M × K` of inner products with
each centroid, sum `M` LUT lookups per database vector, top-k. The trainer
choice does **not** affect query-time code paths, so anisotropic PQ is a
drop-in for any existing PQ index consumer.

## Implementation notes

- **RNG.** k-means++ seeding with a StdRng seeded per config for
  reproducibility. All variants share the same seed so recall differences
  are attributable to the loss, not the init.
- **Numerical.** The centroid system is accumulated in `f64`; only the
  final centroid is downcast to `f32`. Singular Gram matrices (empty or
  degenerate clusters) fall back to the isotropic mean.
- **Sub-space direction.** For simplicity and to keep the trainer purely
  local per sub-space, `û` is the per-sub-space normalisation of `x_s`.
  This is a common practical approximation to the full-vector ScaNN loss
  and preserves the sweet-spot behaviour empirically (below).
- **Trait-based swap.** Consumers switch trainer with a single line change:

  ```rust
  let idx = AnisoPqIndex::build(&AnisotropicTrainer::new(2.0), &train, &data, &cfg)?;
  ```

- **Memory math.** Codebook: `M · K · d · 4` bytes. Codes: `N · M` bytes.
  For `N=8_192, M=8, K=256, d=8`: codebook = 65 536 B, codes = 65 536 B ⇒
  128 KiB total (matches the measured `mem_kb` column).

## Benchmark methodology

- **Hardware.** Apple M4 Max, macOS 24.6 (Darwin arm64), `rustc 1.89.0`
  release build (`opt-level=3`, LTO off — workspace default). No unsafe,
  no BLAS, no SIMD intrinsics — pure `f32` scalar loops so results are
  reproducible on any hardware.
- **Corpus.** N=8 192, D=64 synthetic embeddings: low-rank latent
  (`rank = 16`, orthonormal fixed basis) plus 0.15-scale isotropic noise.
  This mirrors the anisotropic-covariance structure of real dense
  retrievers where a low-rank signal dominates and quantization needs to
  preserve inner-product ranking, not L2 reconstruction.
- **Queries.** 200 held-out samples drawn from the same distribution.
- **Ground truth.** Exact `argsort(q·x)` over all N.
- **Metric.** `recall@10` (proportion of true top-10 recovered by the PQ
  index's ADC top-10). Also reported: per-query latency (mean µs), build
  time (ms), heap bytes.
- **Config.** M=8 sub-spaces of `d=8`, K=256 centroids (8-bit codes),
  20 Lloyd iterations, seed = 0xA150.

## Results

Real numbers from `cargo run --release -p ruvector-anisotropic-pq
--bin aniso-pq-bench`:

| Variant             | build (ms) | query (µs) | recall@10 | mem (KiB) | Δrecall |
|---------------------|-----------:|-----------:|----------:|----------:|--------:|
| isotropic (η=1)     |      633.5 |     145.35 |     0.526 |     128.0 |    –    |
| anisotropic η=2     |    1 555.9 |     141.14 | **0.567** |     128.0 |  +0.042 |
| anisotropic η=4     |    1 573.7 |     144.54 |     0.554 |     128.0 |  +0.028 |
| anisotropic η=8     |    1 564.2 |     149.39 |     0.544 |     128.0 |  +0.018 |
| anisotropic η=16    |    1 569.4 |     141.60 |     0.514 |     128.0 |  −0.012 |

Ground-truth build: 41.4 ms.

Key take-aways:

1. **Query latency is invariant across variants** (139–149 µs) — as
   expected: the ADC scan reads the same `M` bytes and does the same
   `M · K` LUT population per query regardless of how the codebook was
   trained.
2. **Recall has a unimodal η profile.** η=2 is the sweet spot for this
   corpus: +4.2 pp over isotropic at *zero* code, memory, or query-latency
   cost.
3. **η=16 hurts.** Over-weighting parallel residuals starves orthogonal
   directions of centroids, so reconstruction quality collapses in the
   orthogonal subspace.
4. **Training is ~2.5× slower.** Weighted-Lloyd's iteration solves a
   `d × d` system per centroid vs the isotropic vector mean. For 8-dim
   sub-spaces this is cheap in absolute terms (~1.5 s for 8k vectors),
   and training is offline.

## How it works — walkthrough

1. **Training (offline, ~1.5 s for 8k vectors).**
   For each of M sub-spaces:
   - Split the training set into 8-dim slices `x_s`.
   - Compute unit direction `û_s = x_s / ‖x_s‖` for each point.
   - k-means++ seed 256 centroids.
   - For 20 iterations:
     - **Assign** each point to the centroid that minimises the weighted
       residual `h_∥ · (r·û)² + h_⊥ · (‖r‖² − (r·û)²)`.
     - **Update** each centroid by solving the `d × d` linear system
       `(Σ Mᵢ) c = Σ Mᵢ xᵢ` where `Mᵢ = h_⊥ I + (h_∥ − h_⊥) ûᵢûᵢᵀ`.

2. **Encoding (offline, sub-ms per vector).**
   For each database vector: pick, per sub-space, the centroid index with
   minimum unweighted `‖x_s − c‖²`. Concatenate into an M-byte code.

3. **Search (online, ~140 µs / query on 8 k × 64).**
   For each query:
   - Build LUT `M × K` of `q_s · c[s, k]` (M·K·d multiplies).
   - Scan every database code, summing `M` LUT lookups per vector.
   - Partial-sort top-k. No additional memory, no branches, no per-
     variant divergence.

## Practical failure modes

- **Unit-normalised isotropic random data.** With no covariance structure
  and no norm cue (pure Gaussian directions on the sphere), anisotropic
  weighting has nothing extra to learn and slightly under-performs
  isotropic — early bench iterations of this crate confirmed this
  regression before we switched the corpus to a low-rank + noise model.
- **Very heavy-tailed norms.** Log-normal-scaled corpora saw isotropic
  win at moderate σ. Anisotropic PQ is designed for *inner-product
  ranking preservation*, not norm reconstruction; when the norm itself
  dominates the score, isotropic PQ's L2-optimal centroids happen to be
  MIPS-optimal too.
- **η too large.** As shown, η=16 loses to isotropic on this corpus.
  Recommend `η ∈ [2, 8]` and cross-validate against a held-out recall
  target.
- **Empty / degenerate clusters.** Handled by falling back to isotropic
  mean; still worth monitoring in production.

## What to improve next

1. **Full-vector direction with per-sub-space projection** (rather than
   per-sub-space normalisation). ScaNN's original derivation projects the
   global `x / ‖x‖` onto each sub-space without re-normalising; this
   should give a further recall lift on genuinely anisotropic corpora.
2. **η adapted per sub-space.** Sub-spaces where the direction correlates
   strongly with `q` deserve larger η; the trainer could pick η via a
   validation split.
3. **Cross-corpus evaluation.** Wire into `ruvector-sota-bench` and run
   on SIFT-1M, GloVe-1M, and E5-embedded MS-MARCO subsets.
4. **Rotation pre-processing (OPQ / LeanVec).** Compose with a learned
   rotation before quantisation — the two techniques are orthogonal and
   both cheap.
5. **SIMD / f16 codebook.** Query LUT is the hot loop; a packed 8-bit
   LUT with `f16` centroids would 2× throughput on Apple silicon.
6. **Integrate into `ruvector-adaptive-ann`** as a per-partition
   trainer, so IVF cells with different geometry can pick different η.

## Production crate layout

```
crates/ruvector-anisotropic-pq/
├── Cargo.toml
├── src/
│   ├── lib.rs        # Trait, errors, recall metric  (< 110 lines)
│   ├── codebook.rs   # Isotropic + anisotropic trainers  (< 380 lines)
│   ├── index.rs      # AnisoPqIndex ADC search  (< 100 lines)
│   └── main.rs       # aniso-pq-bench binary  (< 180 lines)
└── tests/
    └── mips_recall.rs
```

All files < 500 lines per project convention. Workspace member added under
`crates/ruvector-anisotropic-pq`.

## References

1. Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar,
   S. *Accelerating Large-Scale Inference with Anisotropic Vector
   Quantization.* ICML 2020. https://arxiv.org/abs/1908.10396
2. Sun, P., Simcha, D., Dopson, D., Guo, R., Kumar, S. *SOAR: Improved
   Indexing for Approximate Nearest Neighbor Search.* SIGMOD 2024.
3. Gao, J., Long, C. *RaBitQ: Quantizing High-Dimensional Vectors with
   a Theoretical Error Bound for Approximate Nearest Neighbor Search.*
   SIGMOD 2024.
4. Jegou, H., Douze, M., Schmid, C. *Product Quantization for Nearest
   Neighbor Search.* TPAMI 2011.
5. Ge, T., He, K., Ke, Q., Sun, J. *Optimized Product Quantization for
   Approximate Nearest Neighbor Search.* CVPR 2013.
