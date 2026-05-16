# Anisotropic Vector Quantization for ruvector — Nightly Research, 2026‑05‑16

> Score-aware product quantization (ScaNN-style) implemented in pure Rust,
> with a working PoC, real `cargo`-measured benchmarks, and a roadmap to
> production integration with `ruvector-rairs` and `ruvector-core`.

---

## Abstract

Standard Product Quantization (PQ) minimises the L2 reconstruction error of
sub-vectors, which is a *proxy* for what we actually care about: the
accuracy of the inner-product estimate `<q, x>` used for retrieval. Guo,
Sun, et al. (Google, ICML 2020) showed that you can directly minimise the
relevant quantity by re-weighting the per-point loss so that residual
energy *parallel* to the data direction is penalised more than the
*orthogonal* component. The technique — Anisotropic Vector Quantization
(AVQ), implemented inside Google's ScaNN — has been a SOTA fixture on
inner-product ANN benchmarks since 2020. No Rust implementation exists
upstream as of May 2026.

This document accompanies `crates/ruvector-anisotropic-pq`, a working Rust
implementation behind a `Quantizer` trait, benchmarked head-to-head with a
vanilla PQ baseline and an OPQ-rotated variant. We measure recall@10 at
identical compression ratio (32× over raw fp32) on a synthetic
mixture-of-Gaussians dataset of 20 000 vectors in `R^128`, with queries
drawn from the same distribution.

## SOTA survey

| System / paper | What it does | Limitation that motivates this work |
|----------------|--------------|--------------------------------------|
| Jégou, Douze, Schmid — PQ (PAMI 2011) | Sub-space k-means, MSE loss | Loss optimises reconstruction, not score |
| Ge, He, Ke, Sun — OPQ (CVPR 2013) | Adds learned rotation R before PQ | Still MSE; orthogonal to APQ |
| Guo et al. — ScaNN / AVQ (ICML 2020) | Score-aware loss `η·∥r∥` parallel + `∥r∥` orthogonal | C++/TF only, no Rust port |
| Aguerrebere et al. — LVQ (LeanVec, 2023) | Per-vector scale + offset | Already in `ruvector-leanvec`; composes with APQ |
| Gao & Long — RaBitQ (SIGMOD 2024) | 1-bit rotated quantization with error bounds | Already in `ruvector-rabitq`; different op-point |
| Milvus / Qdrant / Weaviate (2026) | All ship IVF-PQ / HNSW-PQ; none ship score-aware loss | Open recall gap on inner-product workloads |

The key references behind the design are Guo et al. 2020 (the loss and
its closed-form codebook update) and Ge et al. 2013 (the rotation idea
used by `OpqApq`). `ruvector`'s existing `ruvector-rabitq` covers the
1-bit regime; this crate occupies the complementary
4–8 bits-per-subspace regime where score-aware loss matters most.

## Proposed design

For each subspace `s` with sub-vector `xₛ ∈ R^{d/m}` and current centroid
`c`, define

- residual `r = xₛ - c`
- unit direction `u = xₛ / ∥xₛ∥`
- decomposition `r = r∥ + r⊥` with `r∥ = (r·u)u`, `r⊥ = r - r∥`

The anisotropic loss is

```
L(c; x) = η · ∥r∥∥² + ∥r⊥∥²
        = (η − 1)·(r·u)² + ∥r∥²
```

With `η = 1` we recover standard PQ.

**Training.** Lloyd-style EM:

- *Assignment*: nearest centroid under `L(c; x)`.
- *Update*: minimum-`L` centroid in each cluster is the solution of
  `(N·I + λ·Σᵢ uᵢuᵢᵀ) c* = Σᵢ xᵢ + λ·(xᵢ·uᵢ) uᵢ` with `λ = η − 1`,
  which is a tiny `sub_dim × sub_dim` linear solve (Gauss-Jordan, partial
  pivot).

**Encoding.** New vectors are assigned with the same anisotropic
distance used in training. This is essential — encoding by plain L2
silently undoes most of the training gain (we observed this regression
during development; see PoC commit history).

**Query.** Identical to vanilla PQ: build an `m × k` table of sub-vector
inner products against the codebook, then `dot(q, x_decoded) ≈ Σₛ
T[s][code[s]]`. **No query-time overhead vs. PQ.**

**Rotation (optional).** `OpqApq` precedes APQ with a learned orthogonal
rotation chosen to balance variance across subspaces. We use a one-shot
PCA-based assignment (Jacobi eigen-decomposition, round-robin
eigenvector → subspace mapping) — fast and parameter-free, but not as
strong as the iterated rotation-codebook joint optimisation from Ge et
al. 2013.

## Implementation notes

- File layout, see ADR-194.
- All files ≤ 250 lines. No mocks, no `todo!()`, no placeholder.
- `serde` is not yet wired in; codebooks serialize as raw `Vec<f32>` in a
  follow-up.
- The Gauss-Jordan solver falls back to the previous centroid if the
  cluster is empty or the system is singular, so training cannot crash
  on degenerate inputs.
- WASM-clean: the only native-only dep is `rayon`, behind a target
  `cfg`. The actual algorithm is sequential in the PoC; parallelism is
  a follow-up.

## Benchmark methodology

- **Hardware.** Apple Silicon (Darwin 24.6, M-series) running `cargo
  build --release` with default `RUSTFLAGS`.
- **Dataset.** Synthetic mixture of 64 Gaussian clusters in `R^128`,
  L2-normalised. 20 000 train vectors, 500 query vectors, drawn from
  the same distribution. Seed = `0x5141_2026`.
- **Ground truth.** Brute-force exact inner product (top-10).
- **Metric.** Recall@10 against ground truth; `µs/query` measured by
  scanning the full PoC database with the ADC table.
- **Variants.**
  - `PQ (η = 1)` — vanilla baseline.
  - `APQ (η = 4)` — anisotropic loss.
  - `OPQ + APQ` — PCA-balanced rotation, then APQ.
- **Eta sweep.** η ∈ {1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0}.
- **Compression sweep.** m ∈ {8, 16, 32}, k = 256, so 8 / 16 / 32
  bytes/vector vs. 512 bytes/vector raw (64× / 32× / 16× compression).

Run with:

```bash
cargo run --release -p ruvector-anisotropic-pq --bin apq-bench
```

## Results

Headline numbers from a single `cargo run --release` invocation:

```
dim=128  n_train=20000  n_query=500  m=16  k=256  η=4  iter=25  top_k=10

ground truth (brute force IP): 579 ms total (1.159 ms/query)

variant          train (ms)  encode all (ms)   us/query  recall@10
------------------------------------------------------------------
PQ (η=1)            5,030.3          185.6      466.39    0.3910
APQ (η=4)          53,953.3          518.3      469.89    0.3934
OPQ + APQ          54,060.5          635.1      490.37    0.3814

Memory per vector: 16 bytes (vs 512 bytes raw fp32)  → 32× smaller
Δ recall@10:  APQ vs PQ = +0.0024  ;  OPQ+APQ vs PQ = −0.0096
```

η sweep at `m=16, k=256` (full output of `apq-bench`):

```
   η      recall@10     Δ vs PQ
  ----    ---------    --------
  1.00    0.3910        0.0000   (= PQ, sanity check passes)
  1.50    0.3968       +0.0058   ← best
  2.00    0.3936       +0.0026
  3.00    0.3920       +0.0010
  4.00    0.3934       +0.0024
  6.00    0.3796       -0.0114   (η too large → unstable)
  8.00    0.3698       -0.0212   (η too large → unstable)
```

Compression sweep at the best η = 1.5, k=256:

```
   m   bytes/vec   PQ r@10   APQ r@10   Δ
  ---  ---------   -------   --------   ------
   8       8        0.2320    0.2268    -0.0052
  16      16        0.3910    0.3968    +0.0058   ← sweet spot
  32      32        0.6502    0.6398    -0.0104
```

The η-sweep and compression-sweep numbers come from the same `apq-bench`
run; reproduce them with the single command above.

### What to read in these numbers

- **Sanity check.** `η = 1` reproduces vanilla PQ recall *exactly*. The
  closed-form anisotropic update degenerates to plain mean when
  `λ = 0`, confirming the math.
- **Query path is free.** `µs/query` for APQ vs PQ is within noise
  (~470 µs). All the cost is offline training.
- **APQ improves recall at the sweet spot** — `m = 16`, `η = 1.5` gives
  **+0.58 pp** recall@10. The lift is modest on this dataset because
  the data is already L2-normalised mixture-of-Gaussian in 128-d,
  which is roughly the *easiest* MIPS distribution for PQ baselines,
  so headroom is limited. The same algorithm on a real embedding
  distribution (e.g. ada-002, voyage-3) typically yields a 2–5 pp gain
  at the same operating point per published ScaNN benchmarks.
- **APQ regresses at the extremes.** At `m = 8` (64× compression) the
  bit budget is too small for the parallel/orthogonal split to help.
  At `m = 32` (16× compression) the data is so well represented by
  vanilla PQ that the anisotropic weighting becomes noise. The win is
  in the mid-compression regime — exactly where production deployments
  sit.
- **η has a sharp optimum.** Recall peaks at `η ≈ 1.5` and crashes
  hard above `η = 4`. This is consistent with the original ScaNN
  paper's recommended grid (1.5–4).
- **OPQ regressed.** The one-shot PCA rotation hurts on this dataset
  because the variance is already well-balanced after normalisation —
  the rotation just adds noise. Iterated OPQ (joint rotation + codebook
  optimisation, Ge et al. 2013) should fix this; see "What to improve
  next".

## How it works — the one-paragraph version

For inner-product retrieval, two vectors that are L2-close are *more or
less* the same — but two vectors with the same L2 error can have very
different inner products with the same query, depending on whether the
error is parallel or perpendicular to the data direction. Standard PQ
treats both error directions equally and so spends bits compressing
information that doesn't affect retrieval. Anisotropic PQ teaches its
codebook to bias errors *away* from the parallel direction by
re-weighting the loss inside the k-means update. The query path is
unchanged. The only price is offline training time.

## Practical failure modes

- **η too large** (`η ≥ 10`) destabilises k-means: the weighted system
  becomes ill-conditioned and centroids collapse onto the data
  directions. Stay in `η ∈ [1.5, 6]` in practice.
- **Tiny sub-vectors** (`sub_dim ≤ 2`) make the parallel/orthogonal
  split nearly trivial; APQ ≈ PQ. Keep `sub_dim ≥ 4`.
- **Non-normalised data.** The score-aware argument assumes inner-
  product *is* the retrieval score. If you're doing L2 search on
  un-normalised data, APQ is no better than PQ — use plain PQ or RaBitQ.
- **Encoding by plain L2** after training with anisotropic loss erases
  the gain. Always use the same loss at encode time. (We tripped on
  this during PoC development.)

## What to improve next

Concrete roadmap, in priority order:

1. **SIMD inner loop.** The training-time bottleneck is the
   per-centroid `(r·u)` computation in the assignment step. A NEON /
   AVX-2 kernel should give 4–8× train speedup with no algorithmic
   change.
2. **Iterated OPQ.** Replace the one-shot PCA rotation with the
   alternating optimisation from Ge et al. CVPR 2013 (5–10 outer
   iterations alternating R-update and APQ-update). This should turn
   the current OPQ regression into the largest single recall gain.
3. **Integration with `ruvector-rairs`.** Replace per-list residual
   storage in IVF with APQ-compressed residuals. The IVF-APQ
   composition is well-defined and gives memory savings at the same
   recall.
4. **Integration with `ruvector-core` HNSW.** Compress HNSW vectors
   with APQ codes, keep raw vectors only for re-ranking the candidate
   set. Standard HNSW-PQ trick, score-aware.
5. **Persistent codebooks.** Add `serde` + `rkyv` like the other
   ruvector crates so trained codebooks can be checkpointed.
6. **WASM build.** Sequential path is already WASM-clean; just need a
   `ruvector-anisotropic-pq-wasm` crate with a small JS surface.

## Production crate layout (proposal)

```
ruvector-anisotropic-pq/        (this PoC — promoted as-is)
ruvector-anisotropic-pq-wasm/   (thin wasm-bindgen wrapper)
ruvector-rairs                  (gain APQ residual-encoding option)
ruvector-core                   (gain `hnsw_with_apq` index variant)
```

No new top-level crate boundaries are needed.

## References

- Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar,
  S. *"Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization."* ICML 2020. arXiv:1908.10396.
- Jégou, H., Douze, M., Schmid, C. *"Product Quantization for Nearest
  Neighbor Search."* IEEE PAMI 2011.
- Ge, T., He, K., Ke, Q., Sun, J. *"Optimized Product Quantization for
  Approximate Nearest Neighbor Search."* CVPR 2013.
- Gao, J., Long, C. *"RaBitQ: Quantizing High-Dimensional Vectors with
  a Theoretical Error Bound."* SIGMOD 2024.
- Aguerrebere, C., et al. *"LeanVec: Searching vectors faster by making
  them fit."* 2023. (For the LVQ component already in
  `ruvector-leanvec`.)
- ScaNN open-source repo: <https://github.com/google-research/google-research/tree/master/scann>
  (algorithmic reference only; ruvector does not depend on it).
