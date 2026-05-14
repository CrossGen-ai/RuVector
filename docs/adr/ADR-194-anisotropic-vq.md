---
adr: 194
title: "Anisotropic Vector Quantization — ScaNN-style loss-aware PQ for ruvector"
status: accepted
date: 2026-05-14
authors: [ruvnet, claude-flow]
related: [ADR-128, ADR-193]
tags: [quantization, pq, scann, anisotropic, mips, vector-search, nightly-research]
---

# ADR-194 — Anisotropic Vector Quantization (AVQ)

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-14-anisotropic-vq`
as `crates/ruvector-anisotropic-vq`. Build green, 5/5 unit tests pass, demo
binary produces real (non-mocked) numbers via `cargo run --release -p
ruvector-anisotropic-vq --bin avq-demo`.

## Context

ruvector ships a one-bit binary quantizer (`ruvector-rabitq`) and several
IVF/graph indices but **no loss-aware product quantizer**. The
`crates/ruvector-avq` directory existed as an empty placeholder. The 2026
SOTA gap analysis (`docs/research/sota-gap-analysis-2026.md`) flagged
ScaNN-style anisotropic PQ as the largest single compression-vs-recall
improvement available, and ADR-128 listed it as a planned implementation.

The ScaNN paper (Guo et al., ICML 2020, "Accelerating Large-Scale Inference
with Anisotropic Vector Quantization") observes that under maximum inner
product search (MIPS), quantization residuals do not all hurt equally:
the component of the residual **parallel** to the data vector directly
biases the inner-product estimate, whereas the **perpendicular** component
averages out over a uniform query distribution on the sphere. Weighting
the training loss by `eta * ||r_parallel||^2 + ||r_perp||^2` for some
`eta >= 1` therefore aligns the optimisation objective with the ranking
metric. ScaNN reports a 2–3× recall-at-budget improvement over plain
LloydPQ at production scale; we wanted to reproduce that signal in Rust,
without porting any C++ or NumPy code.

## Decision

We introduce `crates/ruvector-anisotropic-vq` exposing a single 8-bit
product quantizer with two training kinds behind a common `QuantizerKind`
enum:

```rust
pub enum QuantizerKind {
    Mse,                            // ordinary Lloyd k-means PQ baseline
    Anisotropic { eta: f32 },       // loss-aware (eta >= 1; eta=1 == Mse)
}
```

Three design choices distinguish this from a naive port:

1. **Closed-form anisotropic centroid update.** For unit-norm full
   vectors and single-subspace PQ replacement, the gradient of the
   anisotropic loss is zero at the solution of a per-cluster linear
   system

   ```text
   A_S = |S| I + (eta-1) sum_{i in S} y_i y_i^T
   b_S = sum_{i in S} y_i + (eta-1) sum_{i in S} ||y_i||^2 y_i
   c*  = A_S^{-1} b_S
   ```

   where `y_i` is the i-th training point's subvector. ds is small
   (typically 4–16), so we solve via partial-pivot Gaussian elimination
   inline (`solve_in_place`, ~30 lines). This is the actual ScaNN
   centroid update, not a heuristic.

2. **MSE encoding given anisotropic codebooks.** Following ScaNN, the
   anisotropic loss decomposes additively across subspaces for
   unit-norm data, so the optimal *encoder* is per-subspace
   nearest-centroid in Euclidean distance — even when codebooks were
   trained anisotropically. An earlier iteration encoded with the
   anisotropic loss directly and was strictly worse than baseline (a
   real, surprising finding we reproduced before fixing); the symmetry
   of distance computation at query time is what makes per-subspace L2
   the right choice.

3. **Pluggable trait-free design.** No `dyn` overhead — `QuantizerKind`
   is a copy enum, `match`ed inside the hot training loop and never at
   query time. The query path is the standard PQ inner-product LUT
   (`build_ip_lut`), unchanged from MSE PQ. Swapping training kinds
   has zero runtime cost on the search side.

## Consequences

### Positive

* **+32% relative recall@1** at K=256 with `eta=3.0` on D=64 Gaussian
  unit-norm vectors at 32× compression (full numbers in the research
  doc). Anisotropic eta=3.0 reaches **19.5% recall@1** vs baseline MSE
  **14.8%** with identical memory footprint and identical query latency.
* **Drop-in for existing PQ users.** Same `encode/decode/build_ip_lut`
  surface as standard PQ.
* **No runtime cost.** All loss-awareness is paid at training time
  (one-time, off-line). Online search is bit-identical to plain PQ.
* **First loss-aware quantizer in ruvector.** Closes the largest
  remaining gap identified in ADR-128 against ScaNN/AQLM/RabitQ.

### Negative

* **Training is 6-8× slower than MSE k-means** for the same data and
  iterations (ds×ds linear solve per cluster per iter). For a 4096-vec
  D=64, M=8, K=256 training run on Apple M4 Max release build: 405 ms
  MSE vs ~2.8 s anisotropic. Acceptable for the off-line phase; a SIMD
  Cholesky would close the gap further.
* **eta is an extra hyper-parameter** and the best value depends on the
  data distribution (we observe eta=2.0–3.0 best on Gaussian).
* **Score-estimate MSE rises slightly** with eta (e.g. 4.07 → 4.12 ×1e-3
  at K=256, eta=3). This is by design: we trade total reconstruction
  fidelity for ranking-relevant fidelity.

### Alternatives considered

* **Optimised Product Quantization (OPQ).** Adds a learned rotation
  before PQ. Complementary to AVQ — could be stacked on top in a
  follow-up. OPQ alone improves perpendicular reconstruction; AVQ
  improves parallel. Combining them is a known SOTA configuration
  ("OPQ + anisotropic loss").
* **Additive Quantization (AQ) / Residual Vector Quantization (RVQ).**
  Higher reconstruction quality but more expensive encoding (greedy or
  beam search, not per-subspace independent). Out of scope for this
  iteration; tracked as a future research topic.
* **AQLM (ICML 2024).** Recent SOTA. Requires solving NP-hard
  assignment per subvector via specialized solvers — significantly
  more engineering than this PoC justifies. Anisotropic loss is the
  strict prerequisite to integrate AQLM later.
* **RaBitQ (already shipped).** Orthogonal: 1-bit quantization for
  rerank scenarios. AVQ targets the 8-bit-per-subquantizer regime where
  RaBitQ cannot compete on recall.

## Notes

* The empty `crates/ruvector-avq/` placeholder is **not** the home for
  this work — we created the new `ruvector-anisotropic-vq` crate to
  keep naming consistent (`ruvector-rabitq`, `ruvector-rairs`, …) and
  to avoid retro-fitting an abandoned scaffold. The placeholder can be
  pruned in a follow-up cleanup.
* No external dependencies beyond `rand 0.8`. No SIMD intrinsics in
  this PoC; the hot training loop is the per-cluster `solve_in_place`
  and the assignment step, both straight scalar Rust. A SIMD pass is
  the obvious follow-up.
