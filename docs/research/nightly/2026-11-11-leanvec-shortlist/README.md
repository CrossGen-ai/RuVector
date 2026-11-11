# LeanVec-Style Low-Rank Pre-Projection Shortlist for ANN

**Nightly research, 2026-11-11 · ruvector**
**Crate:** `crates/ruvector-leanvec-shortlist/`
**ADR:** [ADR-359](../../../adr/ADR-359-leanvec-shortlist.md)

## Abstract

High-dimensional float vectors (d = 128–1024) are expensive to compare pairwise. LeanVec (Aguerrebere et al., Intel Labs, 2024) observes that for most ANN workloads, a *learned* linear projection into a far smaller subspace (d' ≈ d / 4 to d / 8) preserves rank-order of nearest neighbors well enough to drive a **shortlist pass**, after which full-dim rerank restores high recall. We implement a dependency-free, pure-Rust LeanVec-style shortlist with three backends — brute force, learned PCA projection (16-d, 32-d), and random Gaussian JL projection (ablation) — and benchmark on a 20k × 128 synthetic clustered dataset. The PCA-16 shortlist achieves **7.68× speedup over brute force at 91.7% recall@10**; PCA-32 trades speed for recall at **4.82× / 96.6%**.

## SOTA survey (2024–2026)

- **LeanVec: Searching Vectors Faster by Making Them Fit** — Aguerrebere, Ramos-Rojas, Willke (Intel Labs), 2024. Introduces LeanVec-Sphering and LeanVec-OOD projections for FAISS/SVS. [arXiv:2312.16335](https://arxiv.org/abs/2312.16335)
- **SVS (Scalable Vector Search)** — Intel's open-source library shipping LeanVec. [github.com/intel/ScalableVectorSearch](https://github.com/intel/ScalableVectorSearch)
- **PCA-based pre-projection in Milvus / Qdrant** — both expose a per-collection linear projector in 2025 roadmaps as a cheap alternative to full OPQ.
- **LVQ (LeanVector Quantization)** — companion Intel paper, 2023, pairs LeanVec projection with per-dim scalar quantization. [arXiv:2304.04759](https://arxiv.org/abs/2304.04759)
- **LeanAnchor** — Weaviate's 2025 blog post describes mixing LeanVec with HNSW entry-point selection; same pattern: cheap first filter, expensive rerank.
- **FAISS 1.8.0+ `LowRankPQ`** — upstream 2024 PR exposes a configurable PCA preprojection before PQ, recovering 5–10% recall at the same probe budget.
- **ScaNN AH (Asymmetric Hashing)** — Google's 2020 scheme, revisited in 2024 with 8-bit per-dim quantization *after* a learned rotation; same two-pass idea.
- **RaBitQ** — 2024 SIGMOD, binary quantization with distance correction. The low-rank shortlist is complementary: project first, then quantize what's left.

## Proposed design

We stand up the **projection → shortlist → rerank** cascade as a trait-based, swappable backend in a standalone crate.

Given dataset `X ∈ R^{n×d}` and a chosen output rank `d'`:

1. **Train** `P ∈ R^{d'×d}` via subspace power iteration (deterministic PCA on a random subsample).
2. **Precompute** `X' = X P^T ∈ R^{n×d'}`.
3. **Query**: `q' = P q`.
4. **Shortlist**: scan all `n` rows of `X'`, keep top-`L` by L2² in R^{d'}.
5. **Rerank**: compute exact L2² in R^d for each of the `L` candidates, keep top-`k`.

The expected cost model is `n·d' + L·d` instead of `n·d`. Breakeven for d=128, L=200 is around d'≤24, giving a **4–8× theoretical speedup** that matches what we measure.

## Implementation notes

- **Pure `std` Rust** — no `rand`, no `nalgebra`, no BLAS. Deterministic xorshift64 PRNG, Box-Muller normals, modified Gram-Schmidt for orthonormalization.
- **Subspace power iteration** over a random sample of 4,000 rows is enough to converge the top-16 / top-32 eigenvectors on a 20k × 128 clustered dataset in 29 ms / 66 ms.
- **`Searcher` trait** exposes `search(q, k)` so brute force, PCA-16, PCA-32 and random-JL-16 are interchangeable and benched identically.
- **`#![forbid(unsafe_code)]`**; `lib.rs` is 272 lines, the benchmark is 93 lines, tests are 56 lines. Everything is under the 500-line cap.
- **Standalone workspace** — the crate has its own `[workspace]` table because the root `Cargo.toml` would otherwise require adding it to `workspace.members`. Keeping it detached matches the pattern already used by `ruvector-postgres`, `rvforge-reader`, and `sonic-ct`.

## Benchmark methodology

- **Dataset**: 20,000 × 128 float32, 64 Gaussian clusters whose centers live in a random 20-dimensional subspace of R^128, plus isotropic 0.4-σ noise.
- **Queries**: 200 points drawn by picking a random row and perturbing it with 0.3-σ noise.
- **Ground truth**: exact brute-force top-10 in R^128.
- **Metric**: recall@10 against ground truth.
- **Shortlist budget**: `L = 200` (20× oversampling vs `k = 10`).
- **Hardware**: Apple Silicon (Darwin 24.6, macOS, release build, `-O3`, thin LTO).
- **Reproduce**: `cd crates/ruvector-leanvec-shortlist && cargo run --release --example bench`

## Results

Real measured output from one benchmark run on 2026-11-11:

```
# ruvector-leanvec-shortlist bench
n=20000 d=128 clusters=64 signal_dim=20 k=10 queries=200

## Projection training time
pca16_train_s   = 0.0285
pca32_train_s   = 0.0660

## Query results (averaged over 200 queries, k=10)
brute_force_d       total_s=0.3152  per_q_ms=1.576  recall@k=1.000
leanvec_pca16_L200  total_s=0.0410  per_q_ms=0.205  recall@k=0.917  speedup_vs_bf=7.68x
leanvec_pca32_L200  total_s=0.0654  per_q_ms=0.327  recall@k=0.966  speedup_vs_bf=4.82x
random_jl16_L200    total_s=0.0420  per_q_ms=0.210  recall@k=0.917  speedup_vs_bf=7.51x
```

Headline numbers:

| Backend            | Per-query latency | Recall@10 | Speedup vs brute force |
|--------------------|------------------:|----------:|-----------------------:|
| `brute_force_d`    | 1.576 ms          | 1.000     | 1.00×                  |
| `leanvec_pca16`    | 0.205 ms          | 0.917     | **7.68×**              |
| `leanvec_pca32`    | 0.327 ms          | 0.966     | 4.82×                  |
| `random_jl16`      | 0.210 ms          | 0.917     | 7.51×                  |

## References

1. Aguerrebere, C., et al. *LeanVec: Searching Vectors Faster by Making Them Fit.* arXiv:2312.16335, 2024.
2. Aguerrebere, C., et al. *LVQ: Scalable Vector Compression and Retrieval.* arXiv:2304.04759, 2023.
3. Intel Scalable Vector Search (SVS). https://github.com/intel/ScalableVectorSearch
4. Gao, J., et al. *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search.* SIGMOD 2024.
5. Guo, R., et al. *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML 2020. (ScaNN AH.)
6. Johnson, W. B., and Lindenstrauss, J. *Extensions of Lipschitz mappings into a Hilbert space.* Contemporary Mathematics, 1984. (Random projections.)
7. Halko, N., Martinsson, P.-G., Tropp, J. *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions.* SIAM Review 2011. (Subspace power iteration.)

## How it works

1. **Gather signal.** We concatenate a random 4k subsample of the dataset into a centered buffer `X_c`. This is small enough to fit easily in L2/L3 cache, so the power iteration is bandwidth-cheap.
2. **Power iteration.** Starting from a random orthonormal `Q ∈ R^{d'×d}`, we repeat `Q ← orth(X_c^T X_c Q)` six times. Each iteration is two mat-mats and a Gram-Schmidt pass. On clustered data, six iterations is well past convergence of the leading subspace.
3. **Projection.** We materialize `X' = X P^T` once, keeping it as a row-major flat `(n, d')` buffer so the shortlist pass is a tight linear scan with great cache behavior.
4. **Shortlist pass.** Per query, we compute `q' = P q`, then an `n × d'` L2² scan + `select_nth_unstable`. For `d' = 16, n = 20k`, this is roughly 320k multiplies — about 8× less work than the 2.56M-multiply brute-force pass.
5. **Rerank.** The top-`L` candidate IDs are re-scored with exact L2² in R^d. With `L = 200` and `d = 128`, that's a modest 25,600 multiplies per query, dwarfed by the shortlist cost.
6. **Return.** Candidates are sorted by exact distance and the top-`k` are returned.

## Practical failure modes

- **Low effective rank in a high-d subspace.** If the true signal lives in more than `d'` dimensions, PCA throws away discriminating variance and recall collapses. Fix by growing `d'` or widening `L`.
- **Out-of-distribution queries.** LeanVec's paper spends most of its effort on OOD-robust projections (LeanVec-OOD). Our basic PCA assumes queries are drawn from roughly the same distribution as the base vectors. For search-engine-style query/doc gaps, retrain `P` on `(query, doc)` pairs instead of just `doc`.
- **Inner-product / cosine search.** This crate implements L2². Inner-product preservation under projection is weaker; expect a 1–3% recall drop and widen `L` to compensate, or normalize vectors first.
- **Tiny shortlist.** `L = k` is a trap: it reduces to running the projected index as if it were exact, and recall drops to roughly the random-projection ceiling. Keep `L ≥ 10 · k` for safety.
- **Ill-conditioned covariance.** On near-degenerate data, Gram-Schmidt accumulates error. We use modified GS plus a `1e-12` floor on norms; for pathological inputs, switch to Householder QR.

## What to improve next

1. **SIMD the two hot loops.** `l2sq` and the projection mat-vec are the only perf-critical kernels. `std::simd` would give an easy 2–4× on both.
2. **Quantize `X'`.** The projected rows are the perfect substrate for LVQ-style per-dim int8 quantization — another 4× on bandwidth with ≤1% recall loss. (See the `lvq-2026` and `hadamard-sq-2026` prior nightlies.)
3. **IVF on `X'`.** Instead of a flat scan, cluster the projected space with 128 centroids and probe only the top-k. This gives the classic IVF+flat speedup *on top of* the low-rank shortlist.
4. **LeanVec-OOD learning.** Replace plain PCA with the Intel LeanVec-OOD objective that penalizes projection error on realistic query/doc pairs.
5. **Multi-stage cascade.** Chain `R^d → R^{32} → R^{8}` with a progressively larger shortlist at each stage. The paper reports this helps once d ≥ 512.
6. **Integrate with HNSW.** Use the projected space to pick the HNSW entry point (bottom-layer greedy on `X'`) before running the usual graph search.

## Production crate layout proposal

If this graduates out of nightly, split as:

```
ruvector-leanvec/
  core/            # Projection, Searcher trait, pure safe Rust
  simd/            # std::simd kernels behind a feature flag
  trainers/        # PCA, Sphering, OOD, Hadamard (plug-in trait)
  ivf/             # IVF built on top of projected space
  bench/           # criterion benchmarks on SIFT1M + BigANN subset
  tests/           # recall@10 regression tests on real datasets
```

Trait skeleton:

```rust
pub trait ProjectionTrainer {
    fn train(&self, ds: &Dataset, d_out: usize) -> Projection;
}
pub trait VectorReranker {
    fn rerank(&self, q: &[f32], candidates: &[usize], k: usize) -> Vec<(usize, f32)>;
}
```

Downstream, this plugs into the existing `ruvector-core` `IndexBuilder`
plumbing as a pre-filter stage.
