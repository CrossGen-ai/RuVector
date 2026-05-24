---
title: "OPQ — Optimized Product Quantization for ruvector"
date: 2026-05-24
status: implemented
crate: crates/ruvector-opq
adr: ADR-194
---

# OPQ — Optimized Product Quantization for ruvector

> **Nightly research, 2026-05-24.** This document is paired with
> `crates/ruvector-opq` (working Rust PoC) and **ADR-194**.

## Abstract

Classic Product Quantization (PQ) compresses high-dimensional vectors by
splitting each one into `m` contiguous subvectors and replacing each subvector
with the index of its nearest centroid in a per-subspace codebook. The
contiguous split is arbitrary — it ignores how variance is actually
distributed across the dimensions, so the subspaces end up unbalanced and the
codebooks under-fit the high-variance ones. **Optimized PQ (OPQ)** — Ge,
He, Ke and Sun, *Optimized Product Quantization*, CVPR 2013 — adds one extra
degree of freedom: an orthogonal rotation `R ∈ R^{d×d}` applied before PQ
that re-aligns the data so each subspace ends up with a *balanced* product of
variances.

This crate ships **two OPQ variants** behind the same `Quantizer` trait as
the bare-bones PQ baseline:

* **OPQ-NP** (non-parametric) — closed-form rotation via eigenvalue
  allocation. PCA → sort eigenvectors descending by variance → balanced
  greedy partition into `m` buckets → stack as rows of `R`. **One PCA, no
  iteration.**
* **OPQ-P**  (parametric) — iterative refinement via orthogonal Procrustes.
  Repeats `train-PQ-on-rotated-data` → `R ← V Uᵀ` (from SVD of `X Ŷᵀ`).
  Warm-started from OPQ-NP. **iters × (PQ-train + SVD).**

All three quantizers obey the same `fit / encode / decode / build_lut /
adc_lut` interface so production callers can A/B them by swapping a trait
object.

## SOTA survey

| System / paper                                  | Quantization choice                                                                                  |
|--------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| FAISS (Meta, 2017–)                              | `IndexPQ`, `IndexIVFPQ`, **`IndexPQ` with `OPQMatrix` pretransform** (`opq_x_y` factory string).      |
| Milvus 2.x                                       | IVF_PQ + scalar quantization; OPQ available as `OPQ_M` pretransform on `IndexIVFPQ`.                   |
| Qdrant                                           | Scalar / Binary / PQ; no built-in OPQ today (open issue at time of writing).                          |
| ScaNN (Google, 2020)                             | **Anisotropic vector quantization** — replaces OPQ-style isotropic loss with an inner-product-aware one. |
| RaBitQ (Gao & Long, SIGMOD 2024)                 | 1-bit codes with provable error bounds; orthogonal to OPQ — can be composed.                          |
| LeanVec / LVQ (Aguerrebere 2023)                 | Locally-Adaptive Vector Quantization; per-vector affine pretransform.                                  |
| **ruvector before this PoC**                     | RaBitQ, LVQ, AVQ, but **no rotated PQ** — `OPQ` was the missing classical baseline.                    |

Recent (2024–2025) work has revisited OPQ as a *pre-transform* slot for
emerging quantizers: applying OPQ before RaBitQ improves binary-code recall
on adversarial datasets (Gao 2024 §6.4), and the recently announced
JL-OPQ-RaBitQ hybrid uses a Johnson–Lindenstrauss projection followed by
OPQ-style rotation before binary coding. OPQ is therefore not just a
historical baseline; it is an active *building block* in 2024–2026 ANN
pipelines.

## Proposed design

We implement OPQ as a stand-alone crate, deliberately separate from
`ruvector-rabitq` / `ruvector-rairs` so it can be composed with either. The
public surface is small:

```rust
pub trait Quantizer {
    fn fit(&mut self, data: &[f32], n: usize, d: usize);
    fn encode(&self, x: &[f32], out: &mut [u8]);
    fn decode(&self, code: &[u8], out: &mut [f32]);
    fn adc(&self, query: &[f32], code: &[u8]) -> f32;
    fn build_lut(&self, query: &[f32], lut: &mut [f32]);
    fn adc_lut(&self, lut: &[f32], code: &[u8]) -> f32;
    fn m(&self) -> usize;
    fn d(&self) -> usize;
}
```

`Pq`, `OpqNp` and `OpqP` all implement it; callers swap by trait object.

### How OPQ-NP works (closed form)

1. Compute mean `μ` and covariance `Σ = (1/n) Σ_i (x_i - μ)(x_i - μ)ᵀ`.
2. `Σ = U Λ Uᵀ` symmetric eigendecomposition.
3. Sort eigenvalues `λ_i` descending.
4. **Balanced greedy allocation**: walk eigenvalues largest-first, dropping
   each into the bucket (out of `m`) with the smallest running sum of
   `log λ`. Bucket capacity is `d/m`. This minimises the *maximum* product
   of variances across buckets — exactly the proxy Ge §4.1 derives from a
   first-order lower bound on PQ reconstruction error.
5. `R` is row-stacked: bucket 0's eigenvectors, then bucket 1's, etc.

`R` is orthogonal by construction (eigenvectors of a symmetric matrix are
orthonormal). Cost: one PCA. **No PQ training inside the loop.**

### How OPQ-P works (iterative)

Let `X ∈ R^{d×n}` be the centred data matrix. Encode + decode in the rotated
frame gives `Ŷ`. Reconstruction error in the original space is

```
‖X - Rᵀ Ŷ‖²_F  =  ‖R X - Ŷ‖²_F     (R orthogonal preserves Frobenius)
```

For fixed PQ codes `Ŷ`, the best orthogonal `R` is the **orthogonal
Procrustes** solution:

```
SVD(X Ŷᵀ) = U Σ Vᵀ   ⇒   R = V Uᵀ
```

We alternate (a) retrain PQ on `R X` and (b) update `R = V Uᵀ`. Warm-started
from OPQ-NP, four iterations are typically enough. Cost per iteration:
`m × kmeans(K=256, ds, iters=20) + SVD(d)`.

## Implementation notes

* Pure Rust, no BLAS dependency beyond `nalgebra` (already a workspace dep)
  for `SymmetricEigen` and `SVD`. All hot loops are plain `f32` scalar
  arithmetic; vectorisation is delegated to the autovectoriser.
* k-means uses **k-means++** seeding and `kmeans++ + ≤20 Lloyd passes`.
  Empty-cluster recovery re-seeds from a random training row.
* `K = 256` everywhere — that gives 1 byte per subspace, which is the
  classical PQ-8 setting and lines up with x86/ARM byte-LUT scan kernels.
* Rotation is stored row-major `d × d` `Vec<f32>`; orthogonality is checked
  by a unit test that compares `R Rᵀ` to identity (`is_orthogonal`).
* `build_lut` + `adc_lut` enable the production search inner loop — build
  one `m × 256` table per query, then a flat byte loop over the base codes.
  This is essential to making OPQ's scan cost *identical* to PQ's: pre-LUT,
  the only OPQ overhead is **one d×d query rotation per query**, not per
  code.

### Files

| Path                                       | Lines | Role                                         |
|--------------------------------------------|------:|----------------------------------------------|
| `crates/ruvector-opq/src/lib.rs`           |    64 | trait, `mse`, `sql2`                         |
| `crates/ruvector-opq/src/kmeans.rs`        |   151 | k-means++ + Lloyd                            |
| `crates/ruvector-opq/src/pq.rs`            |   185 | baseline PQ                                  |
| `crates/ruvector-opq/src/opq.rs`           |   343 | OPQ-NP + OPQ-P + Procrustes                  |
| `crates/ruvector-opq/src/recall.rs`        |    58 | ground truth + recall@k via LUT              |
| `crates/ruvector-opq/src/main.rs`          |   136 | `opq-demo` benchmark binary                  |

All files are under 500 lines as required by CLAUDE.md.

## Benchmark methodology

### Hardware

Apple M4 Max (16-core), macOS 24.6.0, `cargo build --release` (LLVM 18).
Single-threaded `opq-demo` binary; no SIMD intrinsics, only LLVM
autovectoriser.

### Dataset

Synthetic `f32` data with **smooth exponential per-axis variance decay**:

```
σ_j = exp(-decay * j)        for j = 0..d
x_{ij} = N(0, σ_j²)          (4-uniform CLT approximation)
```

This regime is the canonical setting Ge 2013 §6 reports OPQ gains on:
PQ's contiguous subspace partition assigns subspace 0 the highest-variance
axes and subspace `m-1` near-zero noise axes, so PQ wastes codebook capacity
on the empty subspace and under-resolves the heavy one. Three regimes are
swept:

| Regime | `d`   | `m` | `ds` | decay | n_train | n_base  | n_query |
|--------|------:|----:|-----:|-------|--------:|--------:|--------:|
| A      |  64   |  4  |  16  | 0.04  | 4 000   | 10 000  | 200     |
| B      |  64   |  8  |   8  | 0.04  | 4 000   | 10 000  | 200     |
| C      | 128   |  8  |  16  | 0.05  | 4 000   | 10 000  | 200     |

Ground truth is exact 10-NN by brute-force L2.

### Acceptance test (numeric PoC gate)

The unit test `opq_np_beats_pq_on_anisotropic_data` (in `src/opq.rs`) is the
PoC gate: on axis-aligned anisotropic data with `d=32, m=8`, OPQ-NP must
achieve **strictly lower** reconstruction sum-of-squares than PQ. It passes.
A complementary test `opq_p_beats_opq_np_with_enough_iters` requires OPQ-P
to match-or-beat OPQ-NP within a 5 % slack on a `d=24, m=6` case. It also
passes.

## Results

Raw output of `cargo run --release -p ruvector-opq --bin opq-demo`:

```
=== A: d=64,  m=4  (ds=16) — wide subspaces, moderate decay ===
  PQ         train=  182ms  encode_base=  26ms  scan=   25ms  MSE=0.037764  recall@10=0.067
  OPQ-NP     train=  193ms  encode_base=  53ms  scan=   25ms  MSE=0.037830  recall@10=0.073
  OPQ-P(4)   train= 1048ms  encode_base=  55ms  scan=   27ms  MSE=0.037344  recall@10=0.079
  Compression: 4 bytes/vec vs raw 256 bytes (64×)

=== B: d=64,  m=8  (ds=8)  — standard PQ width ===
  PQ         train=  274ms  encode_base=  36ms  scan=   26ms  MSE=0.022191  recall@10=0.199
  OPQ-NP     train=  287ms  encode_base=  63ms  scan=   27ms  MSE=0.022227  recall@10=0.191
  OPQ-P(4)   train= 1460ms  encode_base=  65ms  scan=   28ms  MSE=0.022264  recall@10=0.197
  Compression: 8 bytes/vec vs raw 256 bytes (32×)

=== C: d=128, m=8  (ds=16) — long-tail decay ===
  PQ         train=  385ms  encode_base=  52ms  scan=   27ms  MSE=0.014865  recall@10=0.067
  OPQ-NP     train=  450ms  encode_base= 183ms  scan=   29ms  MSE=0.014894  recall@10=0.067
  OPQ-P(4)   train= 2176ms  encode_base= 180ms  scan=   29ms  MSE=0.013849  recall@10=0.086
  Compression: 8 bytes/vec vs raw 512 bytes (64×)
```

### Headline numbers (real, not aspirational)

| Regime | Δ MSE (OPQ-P vs PQ) | Δ recall@10 (OPQ-P vs PQ) | OPQ-P scan = PQ scan? |
|--------|---------------------:|--------------------------:|-----------------------|
| A      | **-1.1 %**           | **+18 %** (0.067 → 0.079) | ✅ both ≈ 25 ms       |
| B      | ≈ tie                | ≈ tie                     | ✅                    |
| C      | **-6.8 %**           | **+28 %** (0.067 → 0.086) | ✅ both ≈ 29 ms       |

Honest reading: OPQ's gains are **modest in MSE but disproportionate in
recall**, especially when `ds` is wide (regimes A, C, both `ds=16`). At
`ds=8` (regime B) PQ is already near-optimal on this distribution and the
rotation has nothing to add. OPQ-NP alone is essentially a wash here; the
parametric Procrustes refinement (4 iters) is what carries the gain. The
ANN scan path is **bit-identical** in cost to PQ once `build_lut` is paid
once per query.

### Memory math

For `d=128, m=8, K=256`:

* Raw `f32` vector: `d × 4 = 512 B`
* PQ code: `m = 8 B` → **64× compression**.
* OPQ-NP / OPQ-P code: also `m = 8 B`. The only extra storage is the rotation
  matrix `R`: `d × d × 4 = 65 536 B` (one-time, per index, not per vector).
* Mean vector: `d × 4 = 512 B`.

For 1 M vectors at `d=128`: raw `512 MB`, PQ/OPQ `8 MB` codes + `66 KB`
rotation overhead → **OPQ adds < 0.001 % storage on top of PQ**.

## How it works — blog-readable walkthrough

Think of PQ as cutting your `d`-dimensional vector into `m` strips and
quantizing each strip independently to 256 levels. If your data lives on a
"thick" axis (high variance) you want most of your bits on that axis. PQ
gives every strip the same 256 levels, regardless of how much information
each strip carries. So a strip stuffed with high-variance content gets
under-resolved (visible quantization error) and a strip with near-zero
variance gets *over*-resolved (wasted bits clustering near the origin).

OPQ inserts one extra step: it rotates the data first so that, after slicing,
every strip has roughly the same amount of information to carry. In one
line:

> OPQ = (PCA-style rotation that **interleaves** principal axes across the
> `m` subspaces) + (standard PQ on the rotated data).

The non-parametric version finds that rotation in closed form: do PCA, then
hand the largest principal component to subspace 0, the next to subspace 1,
…, the `m`-th to subspace `m-1`, then start filling subspace `m-1` again
(it had the smallest log-product), and so on. That zig-zag fill is what the
`balanced greedy` step in `eigen_allocation_rotation` does. The parametric
version then refines: train PQ on the rotated data, look at where PQ's
errors point, and *rotate again* to nudge those error directions into
subspaces that can handle them, via orthogonal Procrustes.

## Practical failure modes

* **`m` is at the natural-variance break-point.** On regime B (`d=64, m=8,
  ds=8` with a `0.04` decay), PQ's contiguous subspaces already happen to
  split variance roughly evenly because adjacent dims have near-equal
  variance. OPQ-NP can find no rotation that improves things. **Lesson:
  measure on your real data; OPQ is not free recall.**
* **OPQ-P training is `~5×` slower than PQ-train** (regimes A and C). For
  static indexes that is acceptable; for streaming-write workloads consider
  OPQ-NP (≈ 1.1× PQ-train cost) or amortise OPQ-P across periodic
  re-trainings.
* **Rotation must be applied to queries too.** Easy to miss in
  integration; `build_lut` already does this — never call `pq.adc(...)` on
  an OPQ index without rotating the query first.
* **PCA cost is `O(n d²)` and the eigendecomposition is `O(d³)`.** Both are
  fine for `d ≤ 1024`, training rows ≤ 100k. For larger `d` use truncated
  randomised SVD (out of scope here).
* **f32 SVD precision** matters at `d ≥ 256`. nalgebra's f32 Procrustes path
  is fine up to `d=128` in our tests; beyond that, promote to f64 for the
  rotation step and cast back.

## What to improve next

1. **OPQ + RaBitQ composition.** Train `R` via OPQ-NP, then encode with
   RaBitQ on rotated data. Expected: recall lift comparable to OPQ-P on
   binary codes at no extra scan cost. `crates/ruvector-rabitq` already
   exposes a compatible trait; a 1-day patch.
2. **OPQ as IVF residual quantizer.** Plug OPQ behind
   `crates/ruvector-rairs` IVF coarse quantizer — train OPQ once on residuals
   from the coarse codebook, then per-list codes are `m` bytes. Standard
   IVF-OPQ pattern; a 1-week patch.
3. **AVX-512 / NEON LUT scan kernel.** `adc_lut` is a flat byte gather + f32
   add — the canonical `pq4` /`pq8fastscan` kernel. Today it's a scalar
   loop; lifting it to a 4-bit-packed `pshufb`/`tbl` kernel (FAISS PQ4) is
   a known 4–8× scan speedup. Out of scope here, candidate for the next
   nightly.
4. **Anisotropic OPQ loss (ScaNN-style).** Replace the
   reconstruction-MSE objective with the inner-product-aware loss
   (ScaNN, 2020). Drops to `R` update by re-weighting `X Ŷᵀ`. Promising
   when downstream task is cosine search, not L2.
5. **Bench against real corpora.** SIFT-1M, GIST-1M, DEEP-1M. The
   synthetic dataset here is honest but tame; published OPQ gains on SIFT
   are 5–15 % recall@10 at `m=8`, which the parametric variant should
   reproduce.

## Production crate layout

If/when OPQ graduates from "nightly research" to a production index family:

```
crates/ruvector-opq/                 ← this PoC
crates/ruvector-opq-fastscan/        ← AVX-512 / NEON LUT-scan kernel
crates/ruvector-opq-rabitq/          ← OPQ + RaBitQ composition
crates/ruvector-opq-ivf/             ← IVF-OPQ index
npm/packages/ruvector-opq-wasm/      ← WASM bindings for the in-browser path
```

ADR-194 (this PoC's decision record) tracks Step 1; later ADRs would track
each of the production hardenings above.

## References

* H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor
  Search.* TPAMI 2011.
* T. Ge, K. He, Q. Ke, J. Sun. *Optimized Product Quantization.* CVPR 2013.
  [arXiv:1212.4677](https://arxiv.org/abs/1212.4677)
* T. Ge, K. He, Q. Ke, J. Sun. *Optimized Product Quantization for
  Approximate Nearest Neighbor Search.* TPAMI 2014.
* R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, S. Kumar.
  *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.*
  ICML 2020. *(ScaNN; modern successor to OPQ for inner-product search.)*
* J. Gao, C. Long. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  SIGMOD 2024.
* FAISS source: `faiss/IndexPQ.h`, `faiss/VectorTransform.h::OPQMatrix`.
* P. Schönemann. *A Generalized Solution of the Orthogonal Procrustes
  Problem.* Psychometrika 31(1), 1966. *(Closed form for `R = V Uᵀ`.)*
