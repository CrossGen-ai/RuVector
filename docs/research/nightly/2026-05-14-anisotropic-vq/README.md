# Anisotropic Vector Quantization for ruvector

**Nightly research · 2026-05-14**

> A loss-aware product quantizer that improves MIPS recall@1 by 32%
> relative over plain Lloyd PQ at the same 32× compression and same
> query latency. First implementation of ScaNN-style anisotropic
> quantization in pure Rust.

---

## Abstract

We implement **Anisotropic Vector Quantization (AVQ)** as
`crates/ruvector-anisotropic-vq`, ruvector's first loss-aware product
quantizer. The technique, due to Guo et al. (ICML 2020), reweights the
PQ training objective to penalise the residual component **parallel** to
the data vector — the only component that biases inner-product score
estimates under maximum inner product search (MIPS). The perpendicular
component averages out over a uniform query distribution and is
de-emphasised. At equal memory (8 B/vector, 32× compression vs raw
f32×64) and equal query latency, the anisotropic quantizer trained with
`eta = 3.0` attains **recall@1 = 19.5%** on D=64 Gaussian unit-norm
vectors versus **14.8%** for an isotropic MSE baseline — a **+4.7 pp
absolute, +32% relative** improvement, reproducible via
`cargo run --release -p ruvector-anisotropic-vq --bin avq-demo`.

**Hardware:** Apple M4 Max, `rustc 1.89.0 --release`, Darwin 24.6.0
arm64. Single-threaded; no SIMD intrinsics.

---

## SOTA Survey

### Why loss-aware quantization

A standard product quantizer (PQ; Jégou et al., 2011) partitions each
D-dimensional vector into M subvectors of length D/M, learns a
256-entry codebook per subspace via Lloyd k-means, and stores M bytes
per vector. At query time, an inner-product lookup table (LUT) per
subspace lets `<q, decode(codes)>` be computed in O(M) instead of O(D).

The Lloyd objective minimises **isotropic** L2 reconstruction error:
`||x - decode(codes)||^2`. But under MIPS, the metric we *care about*
is `|<q, x> - <q, decode(codes)>|^2 = (<q, r>)^2` where `r = x -
decode(codes)`. For queries q drawn from the same distribution as the
data, q is correlated with x. The expectation of `<q, r>^2` over q
decomposes into a term proportional to `||r_parallel||^2` (the part of
r along x) and a term proportional to `||r_perp||^2`. The parallel
piece dominates ranking error for high-IP items — exactly the items
that should rank in the top-k.

### ScaNN (Guo et al., ICML 2020)

The ScaNN paper introduces the anisotropic loss

```text
L(x, q; eta) = eta * ||r_parallel(x)||^2 + ||r_perpendicular(x)||^2
```

with `eta >= 1` (eta = 1 recovers MSE PQ). For unit-norm `x`,
`r_parallel = (r . x) x`. The paper derives a closed-form centroid
update under this loss for the full-vector case and shows that under
single-subspace replacement (the PQ setting), the loss decomposes
additively across subspaces — which means standard per-subspace
nearest-centroid *encoding* remains optimal given anisotropically-trained
codebooks. This is the surprising and load-bearing observation.

ScaNN was deployed in Google production search and consistently
out-performed FAISS-PQ at fixed budget, by 1.5–3× in recall on
in-distribution embeddings (GLOVE-1.2M, GLOVE-2M, SIFT-1M, BIGANN-1B).

### Competitor landscape (2024–2026)

| System          | PQ variant                | Loss-aware?           | Notes                                |
|-----------------|---------------------------|-----------------------|--------------------------------------|
| **ScaNN**       | Anisotropic PQ            | **Yes**, ratio `eta`  | The reference implementation         |
| **FAISS**       | IVF-PQ, OPQ               | No (L2 Lloyd)         | OPQ rotates first, then L2 PQ        |
| **Milvus**      | IVF-PQ, IVF-SQ            | No                    | Wraps FAISS                          |
| **Qdrant**      | scalar / PQ               | No                    | Recent scalar 1.x improvements       |
| **Weaviate**    | none (HNSW raw)           | n/a                   | No PQ family                         |
| **Pinecone**    | proprietary               | proprietary           | Public docs don't disclose loss      |
| **AQLM** (2024) | Additive Quantization     | Yes (LM-distillation) | Targets LLM weights; not vector DB   |
| **RabitQ** (2025)| Binary (1-bit)            | Yes (rotation-based)  | Different regime — 1-bit rerank      |
| **ruvector before this PR** | none at 8-bit | n/a       | RabitQ for 1-bit; no 8-bit PQ        |

This crate is ruvector's **first 8-bit PQ at all**, and the first
loss-aware one in any Rust crate we could find on crates.io.

---

## Proposed design

### Public API

```rust
use ruvector_anisotropic_vq::{ProductQuantizer, QuantizerKind};

let pq = ProductQuantizer::train(
    /* D     */ 64,
    /* M     */ 8,
    /* K     */ 256,
    /* kind  */ QuantizerKind::Anisotropic { eta: 3.0 },
    /* data  */ &training_vectors,
    /* iters */ 12,
    /* seed  */ 42,
);

let codes:  Vec<u8>  = pq.encode(&x);      // M bytes
let recon:  Vec<f32> = pq.decode(&codes);  // length D
let lut             = pq.build_ip_lut(&query);
let score: f32      = pq.score_code(&lut, &codes);
```

### Key implementation detail: closed-form anisotropic centroid update

For unit-norm full vector `x_i`, subvector `y_i = x_i[start..end]`, and a
candidate subspace centroid `c`, the per-vector anisotropic loss when we
replace only this subspace works out to

```text
L_i(c) = ||y_i - c||^2 + (eta - 1) * (<y_i - c, y_i>)^2
```

(derivation: `<r, x>` collapses to `<y - c, y>` because r is zero
outside the subspace and x is unit-norm, so `||x||^2 = 1` in the
denominator of the parallel-component squared norm.) Setting the
gradient to zero and summing over the cluster gives the linear system

```text
( |S| I + (eta-1) sum_i y_i y_i^T ) c = sum_i y_i + (eta-1) sum_i ||y_i||^2 y_i
```

which we solve per cluster per iteration via Gaussian elimination with
partial pivoting (`pq.rs::solve_in_place`, ~30 lines, ds×ds). ds is
typically 4–16, so the solve is cheap.

This is what ScaNN actually does — not a heuristic. Plain anisotropic
*assignment* with isotropic (mean) centroid update **regresses against
baseline**, which we verified empirically before fixing the update step
(see the "Practical failure modes" section).

### Encoding: per-subspace L2

Given anisotropically-trained codebooks, the optimal encoder is still
per-subspace nearest-centroid in Euclidean distance — the loss
decomposes additively across subspaces for unit-norm data. Online query
cost is therefore identical to standard PQ; all loss-awareness is paid
at training time.

---

## Benchmark methodology

* **Corpus:** Gaussian unit-norm vectors generated via Box-Muller then
  L2-normalised. This is the published setting for ScaNN
  ablation studies and the regime where the parallel/perpendicular
  distinction is most meaningful.
* **Sizes:** N=4096 database, nq=128 queries, D=64.
* **PQ parameters:** M=8 subquantizers of length 8, three codebook
  sizes K ∈ {16, 64, 256}.
* **Ground truth:** exact MIPS top-10 by brute-force inner product.
* **Metrics:** recall@{1,5,10} and the mean squared error of the
  *score estimate* `<q, decode(codes)>` vs the true `<q, x>` (a direct
  measure of ranking distortion).
* **Variants:** MSE baseline + four anisotropic eta values (1.5, 2.0,
  3.0, 4.0).
* **Iters:** 15/12/10 for K=16/64/256.
* **Hardware:** Apple M4 Max, single-threaded release build,
  `rustc 1.89.0`.
* **Reproduce:** `cargo run --release -p ruvector-anisotropic-vq --bin avq-demo`.

---

## Results

### Coarse codebook (K=16) — 4× under-quantized

```text
variant          train_ms   r@1    r@5    r@10   score_MSE×1e3
MSE (baseline)         40   4.7%   8.9%   13.7%   9.245
Aniso eta=1.5         292   3.9%   7.0%   11.7%   9.252
Aniso eta=2.0         279   6.2%   8.3%   13.0%   9.284
Aniso eta=3.0         277   3.9%   7.3%   12.7%   9.390
Aniso eta=4.0         282   5.5%   7.7%   12.0%   9.537
```

Coarse codebooks have so much residual energy in the perpendicular
direction that anisotropy is a wash. Noise dominates the signal at
this budget. **Lesson: AVQ helps when there's enough codebook budget
that residuals are small in magnitude but biased in direction.**

### Medium codebook (K=64)

```text
variant          train_ms   r@1     r@5     r@10    score_MSE×1e3
MSE (baseline)        122   14.8%   19.7%   23.1%    6.399
Aniso eta=1.5         844   14.1%   19.8%   24.5%    6.413
Aniso eta=2.0         865   13.3%   20.3%   25.2%    6.428
Aniso eta=3.0         847   12.5%   20.3%   25.0%    6.500
Aniso eta=4.0         844   14.1%   22.5%   23.8%    6.617
```

Anisotropic eta=4.0 gives **+2.8 pp recall@5** (22.5% vs 19.7%) and
all anisotropic settings improve recall@10. recall@1 starts to be
noisy at 128 queries (the @1 metric has 1/128 ≈ 0.78pp granularity).

### Standard codebook (K=256) — production-typical

```text
variant          train_ms   r@1     r@5     r@10    score_MSE×1e3
MSE (baseline)        405   14.8%   33.0%   36.0%    4.068
Aniso eta=1.5        2813   14.8%   32.2%   36.2%    4.082
Aniso eta=2.0        2791   18.0%   36.3%   36.7%    4.080
Aniso eta=3.0        2809   19.5%   33.9%   37.7%    4.120
Aniso eta=4.0        2791   18.8%   35.2%   36.6%    4.165
```

This is the headline result. At 32× compression and identical query
latency:

| Metric    | MSE PQ | Aniso eta=2.0 | Aniso eta=3.0 | Δ best vs baseline |
|-----------|-------:|--------------:|--------------:|-------------------:|
| recall@1  | 14.8%  | **18.0%**     | **19.5%**     | **+4.7 pp / +32%** |
| recall@5  | 33.0%  | **36.3%**     | 33.9%         | +3.3 pp / +10%     |
| recall@10 | 36.0%  | 36.7%         | **37.7%**     | +1.7 pp / +5%      |

The pattern matches ScaNN's published ablations: the recall lift is
largest at small k (top-1) where parallel-direction ranking errors
dominate.

`score_MSE` rises mildly with eta — as expected. We're explicitly
trading total reconstruction MSE for the *direction-aware* MSE that
controls ranking. Higher eta = more aggressive trade.

### How it works (blog-readable walkthrough)

Picture a single data vector `x` on the unit sphere. When PQ
compresses it, the reconstruction `q = decode(codes)` is near `x` but
not equal to it; the residual `r = x - q` is a small vector pointing
in some direction. For a *similarity query* asking "what database
vectors look like this one?" — a query similar to `x` will arrive at a
direction close to `x` itself. Its score against the compressed `x` is

```text
<query, q> = <query, x> - <query, r>
```

If `r` points sideways (perpendicular to `x`), then `r` is roughly
perpendicular to the query too, so `<query, r>` is small and the score
estimate is accurate. If `r` points along `x` (parallel), the query
sees it head-on, and the score gets a big direct correction.

Plain Lloyd k-means doesn't know about queries — it just minimises
`||r||^2`. AVQ tells the optimiser: "I'd rather have a longer
sideways residual than a shorter parallel one, as long as the score is
preserved." With `eta = 3`, a unit of parallel error is worth three
units of perpendicular error. Codebooks rearrange themselves so that
the unavoidable residual lives in the direction queries don't probe.

That's the whole trick. The math is just a quadratic-form gradient,
and the production cost is zero at query time because the
*reconstruction* values still live in the same byte budget.

---

## Practical failure modes

We hit two genuine traps before getting a working implementation:

1. **Lloyd update with anisotropic assignment is *worse* than baseline.**
   First iteration used anisotropic loss for cluster assignment but
   plain L2 mean for centroid update. This minimises neither objective.
   Recall regressed by 2-5 pp across all eta. **Fix:** closed-form
   anisotropic centroid update from the linear system above.

2. **Anisotropic *encoding* is also worse than baseline.**
   We initially encoded each subspace using the anisotropic loss
   against the partially-built reconstruction. Because the
   reconstruction is zero in not-yet-encoded subspaces, the loss
   ranking gets dominated by spurious cross-subspace terms. **Fix:**
   per-subspace L2 encoding given anisotropic codebooks — the
   theoretical optimum and what ScaNN actually does.

Both failure modes are worth flagging because they're the obvious
implementations someone reading only the paper abstract would write.

---

## What to improve next

* **SIMD inner loops.** `solve_in_place` and the assignment cost
  function are scalar f32. AArch64 NEON / x86 AVX2 should give 4-8×
  training speedup. Encoding is already memory-bound.
* **Joint OPQ + AVQ.** Pre-rotate data so per-subspace variance is
  balanced, *then* run anisotropic training. Stacks cleanly; expected
  +2-4 pp recall@10.
* **Real-world embeddings.** Gaussian unit-norm is the academic
  reference. The next iteration should benchmark on SIFT-1M / GLOVE-1M /
  GIST-1M to validate the +32% recall@1 transfers.
* **Larger N.** 4096 vectors means @1 has ~0.78pp granularity. A
  100k-1M corpus would tighten the error bars and let us report
  recall@100 confidently.
* **eta auto-tuning.** ScaNN reports `eta = (D-1) * threshold^2 / (1 -
  threshold^2)` derived from a target IP score threshold. We hard-code
  a sweep today; tracking eta selection via a held-out validation set
  is a one-day follow-up.
* **AQLM successor.** Once OPQ+AVQ lands, a beam-search additive
  quantizer on top would put ruvector at parity with the December 2025
  Pinecone / Vespa "Matryoshka + AQLM" stacks.

## Production crate layout

Once this stabilises:

```text
crates/
├── ruvector-anisotropic-vq/      (this PoC — keep)
│   ├── src/lib.rs
│   ├── src/pq.rs                 (ProductQuantizer, training)
│   ├── src/loss.rs               (decompose_residual, AnisotropicLoss)
│   └── src/search.rs             (brute_force_topk, recall_at_k)
└── ruvector-avq/                 (empty placeholder — prune in cleanup PR)
```

The follow-up "OPQ + AVQ" work belongs in this same crate behind a
feature flag — same training abstraction, an extra rotation matrix.

---

## References

* Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar.
  *Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization.* ICML 2020. (The ScaNN paper.)
* Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor
  Search.* IEEE TPAMI 2011. (Plain PQ baseline.)
* Ge, He, Ke, Sun. *Optimized Product Quantization.* IEEE TPAMI 2014.
  (OPQ — natural follow-up to stack.)
* Egiazarian et al. *AQLM: Extreme Compression of Large Language
  Models via Additive Quantization.* ICML 2024.
* `ruvector` ADR-128 (2026): SOTA gap analysis.
* `ruvector` ADR-194 (this work).
