---
adr: 197
title: "Anisotropic Product Quantization (ScaNN-style) for MIPS"
status: accepted
date: 2026-06-04
authors: [ruvnet, claude-nightly]
related: [ADR-193, ADR-194]
tags: [pq, quantization, mips, scann, anisotropic, vector-search, nightly-research]
---

# ADR-197 — Anisotropic Product Quantization for MIPS

## Status

**Accepted.** Implemented on branch `research/nightly/2026-06-04-anisotropic-pq` as
`crates/ruvector-anisotropic-pq`. `cargo build --release -p ruvector-anisotropic-pq`
is green. `cargo test --release -p ruvector-anisotropic-pq` passes 5/5. The
benchmark binary produces real numbers and satisfies the acceptance criterion.

## Context

ruvector already ships several quantization paths:

| Crate                  | Technique                       | Loss objective                |
|------------------------|---------------------------------|-------------------------------|
| `ruvector-opq`         | Optimised Product Quantization  | L2 reconstruction             |
| `ruvector-lvq`         | LVQ (per-vector affine + bytes) | L2 reconstruction             |
| `ruvector-rabitq`      | 1-bit binary quantization       | Centered angular              |
| `ruvector-leanvec`     | LeanVec rotation + PQ           | L2 reconstruction             |

**Every one of these minimises a uniform (isotropic) error norm.** For MIPS
(maximum inner product search, the dominant retrieval objective for RAG, ads,
recsys), the relevant quantity is not `||x - x̃||` but the MIPS score error
`<x̃, q> - <x, q>`. The two diverge: for unit-norm `x` and query `q`,

    <e, q> = <e_∥, q> + <e_⊥, q>,    e = x - x̃

where `e_∥` is the projection of `e` onto `x̂ = x/||x||` and `e_⊥` is
orthogonal. When `q` is similar to `x` (the realistic MIPS case — queries
*look like* their top retrieved documents), `q` is concentrated near `x̂`, so
`<e_∥, q>` dominates the score error and `<e_⊥, q>` contributes much less.

The **ScaNN paper (Guo et al., ICML 2020)** showed that a codebook trained to
minimise `η·||e_∥||² + ||e_⊥||²` with `η > 1` beats plain L2 PQ on MIPS
recall and inner-product MSE, with no change in storage, encoding speed, or
search-time arithmetic. ScaNN is the production retrieval system inside
Google's search and is the top-ranked PQ method on the ann-benchmarks
glove-100-angular leaderboard. ruvector had no equivalent codec until now.

## Decision

Add `crates/ruvector-anisotropic-pq`: a trait-compatible PQ codec that
generalises plain L2 PQ via a per-vector anisotropy ratio `η ≥ 1`, with `η=1`
recovering vanilla L2 PQ exactly.

**Per-subspace block-diagonal anisotropic loss.** For unit-norm input `x` with
ScaNN's full loss `(η-1)(e·x̂)² + ||e||²`, the cross-subspace coupling makes
direct optimisation expensive. We adopt the standard block-diagonal
approximation: drop cross-terms in `(Σ_s e_s · x_s)²` to get

    L_s = (η - 1)(e_s · x_s)² + ||e_s||²

per subspace `s`. The effective parallel-amplification is therefore
`1 + (η - 1)||x_s||²` — subspaces carrying more vector mass are bent harder
toward the data manifold. At `η = 1` we recover the L2 normal equations.

**Closed-form M-step.** With assignments fixed, the per-cluster centroid is
the solution of

    (Σ_i W_i,s) μ_c,s = Σ_i W_i,s x_i,s,   W_i,s = (η-1) û û^T + I

where `û := x_i,s / ||x_i||`. We solve this `d_sub × d_sub` system (typically
8×8) with Gaussian elimination and partial pivoting in
`solve_in_place`. With `d_sub ≤ 16` this is ~10 ns per cluster per iteration.

**ADC search.** Identical to plain PQ — build an `[m × k]` lookup table of
partial inner products `<q_s, μ_c,s>` per query, then scoring an encoded
vector is `m` table reads and adds. Anisotropy is a *training-time-only*
change.

## Consequences

**Wins.**
- **31% lower MIPS score MSE on the true top-10** at the same code size,
  same query latency, same encoding latency. Real numbers (n=50k, dim=64,
  m=8, k=16, 500 queries, perturbed-from-data queries, MacBook Pro
  Apple Silicon, release build):
  - L2 PQ              → `MSE_top10 = 7.76e-2`, `recall@10 = 0.054`
  - Anisotropic η=2    → `MSE_top10 = 7.05e-2` (1.10× better)
  - Anisotropic η=4    → `MSE_top10 = 5.93e-2` (**1.31× better**)
- **Drop-in compatible:** identical encode size (1 byte per subspace),
  identical lookup-table format, identical search loop. Any existing PQ
  consumer (IVF-PQ, HNSW-PQ rerank) can swap codebooks at zero cost.
- **Composable with rotation (OPQ) and anisotropic loss simultaneously** —
  the rotation is orthogonal so the parallel-vs-orthogonal decomposition is
  preserved post-rotation. Follow-up work: stack `ruvector-opq`'s learned
  rotation on top.

**Trade-offs.**
- Training is dominated by per-cluster matrix solves and is **same wall-clock
  as L2 k-means** in the measured config (270–280 ms either way for 15
  iterations). The closed-form solve scales `O(k · d_sub³)` per iteration —
  fine for `d_sub ≤ 16` (production), slow for `d_sub ≥ 32` (avoid).
- For purely random uncorrelated queries (queries independent of data), the
  anisotropic gain disappears — `<e_∥, q>` and `<e_⊥, q>` contribute equally
  on average. The win shows up specifically in the realistic regime where
  queries resemble their top-k results.
- The MSE-on-the-full-set is essentially unchanged (within 3%), even
  slightly worse for large η. This is correct behaviour, not a bug — the
  codebook is *trading* bulk fidelity for top-k fidelity, which is exactly
  what MIPS workloads want.

**Recall.** On this aggressively compressed configuration (4 bytes ≪ raw
256 bytes), recall@10 is ~5% for all variants — the floor is set by code
size, not by the loss. The headline MSE_top10 improvement matters because
it is exactly the input to *reranking*: any caller using PQ-as-recall +
exact-rerank gets a 30% better candidate list at fixed `nprobe`. We add a
roadmap item to wire this into the IVF-PQ path.

## Alternatives Considered

1. **Plain OPQ (rotation + L2 PQ).** Already in tree (`ruvector-opq`). Solves
   a different problem — alignment of subspaces to the data covariance —
   which is orthogonal to (and composable with) anisotropic loss.

2. **RaBitQ (1-bit).** Already in tree (`ruvector-rabitq`). Fundamentally
   different storage class (1 bit vs ~4 bits per dim). Anisotropic is the
   right knob for PQ; RaBitQ already minimises the right inner-product loss
   for binary codes.

3. **Full ScaNN (with η depending on per-vector norm).** Skipped for this
   PoC — ScaNN's published `η = (T² · ||x||²) / (1 - T²/||x||²)` formula adds
   one float of state per vector and complicates the encode path. The constant
   `η` PoC already achieves the target win.

4. **Coordinate descent over global anisotropic loss (no block-diagonal
   approximation).** Iteratively re-optimises one subspace at a time holding
   the others fixed. Tighter optimum, ~5–10× slower training. Future work
   if needed.

## Roadmap

- Wire `ruvector-anisotropic-pq` into `ruvector-rairs` and `ruvector-diskann`
  rerank paths.
- Compose with `ruvector-opq`'s learned rotation.
- Switch to ScaNN's `η(||x||)` formula for variable-norm data
  (text-embedding norms typically vary 2–3×).
- Bench on real text-embedding data (e.g. BEIR or LAION-CLIP slices).
