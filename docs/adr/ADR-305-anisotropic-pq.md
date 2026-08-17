# ADR-305 — Anisotropic (score-aware) Product Quantization

- **Status**: Proposed (research spike)
- **Date**: 2026-08-17
- **Nightly research**: `docs/research/nightly/2026-08-17-anisotropic-pq/`
- **Crate**: `crates/ruvector-anisotropic-pq/`
- **Related**: ADR-035 (baseline PQ), ADR-303 (entropy-adaptive ANN), the
  `ruvector-pq-search` / `ruvector-rabitq` crates.

## Context

Standard Product Quantization (PQ) trains each subquantizer's codebook via
Lloyd's k-means on the L2 reconstruction error `||x − x̂||²`. For maximum
inner-product search (MIPS) — the dominant retrieval workload for LLM
embeddings, recommenders, and RAG — L2 error is a poor proxy for what
actually harms recall: **error components aligned with the query direction
distort dot-product scores more than perpendicular errors of equal magnitude.**

Guo et al. (ICML 2020, "Accelerating Large-Scale Inference with Anisotropic
Vector Quantization") introduced a score-aware loss:

```
L_aniso(x, x̂) = h_∥ · ||r_∥||² + h_⊥ · ||r_⊥||²        with r = x − x̂
```

that penalizes parallel error more heavily and drives their ScaNN library's
recall advantage on ANN-benchmarks. This has never been implemented in
ruvector; every existing PQ variant (`ruvector-pq-search`,
`ruvector-rabitq`, `ruvector-turboquant`) uses either L2 loss or bitwise
alternatives.

## Decision

Land a research PoC crate `ruvector-anisotropic-pq` that implements **three
swappable PQ trainers** behind a shared `PqTrainer` trait:

1. **`L2Trainer`** — baseline L2 Lloyd's k-means (reference).
2. **`NormWeightedTrainer`** — "score-aware lite": each vector contributes
   to k-means updates with weight `||x||²`, biasing quantization toward
   high-norm vectors (which dominate MIPS scores).
3. **`AnisotropicTrainer`** — the paper's loss, applied at the subvector
   level with a **closed-form d_sub × d_sub linear solve per centroid**
   (Gauss-Jordan) for the update step:

   ```
   A c = b   with   A = Σᵢ (I + (η−1) uᵢ uᵢᵀ),
                    b = Σᵢ (I + (η−1) uᵢ uᵢᵀ) sᵢ,
                    uᵢ = sᵢ / ||sᵢ||
   ```

Ship a real `cargo run --release` benchmark harness that reports
recall@10 vs exact inner-product ground truth on synthetic heavy-tail-norm
data (Pareto-scaled Gaussian), plus training/encoding/query cost.

## Consequences

**Positive**
- Adds a first-class score-aware PQ path to the workspace without touching
  existing PQ crates.
- The `PqTrainer` trait provides a clean insertion point for future
  variants (OPQ, LSQ, RaBitQ hybrids).
- Real benchmark numbers guide which variant to promote:
  - **NormWeighted delivers +10.6–10.7 pp recall@10 over L2-PQ** at
    identical memory footprint and identical training cost — this is a
    clean win for production embedding stores with heavy-tail norm
    distributions.
  - Full anisotropic (per-subvector) **hurts recall** by −2 to −9 pp and
    costs ~2.7× training time — a valuable **negative result** documented
    in the research doc.

**Negative**
- Anisotropic training is 2.7× slower than L2 due to per-centroid
  `d_sub × d_sub` linear solve.
- The per-subvector loss is a simplification of the paper's full-vector
  loss; the loss does *not* factorize exactly across subquantizers, and
  our benchmarks quantify the resulting quality regression.
- Adds one more workspace member (compile-time cost).

**Risk**
- If a downstream consumer picks `AnisotropicTrainer` on real data
  expecting a ScaNN-style win, they will see a recall regression. The
  crate docs and this ADR must call this out explicitly.

## Alternatives considered

1. **Port ScaNN's full-vector loss with joint OPQ rotation** — proper
   implementation. Rejected for this spike: needs a d × d rotation matrix
   fitted jointly with codebooks (Riemannian optimization on SO(d)) and
   is ≥ 2× the scope. Documented as roadmap in the research doc.

2. **Extend `ruvector-pq-search` in-place** — rejected: mixing three
   loss variants in one crate would tangle the ADC scoring path and
   invalidate that crate's existing benchmarks. A standalone crate lets
   us benchmark cleanly and promote whichever variant proves best.

3. **Skip anisotropic entirely, ship only norm-weighted** — rejected: the
   negative result on per-subvector anisotropic loss is itself the
   contribution, and the closed-form d_sub × d_sub solve is reusable for
   the full-vector rotation follow-up.

## Follow-up (out of scope for this ADR)

- **Full-vector anisotropic loss + OPQ rotation** — the honest path to
  matching ScaNN's numbers. Tracked as "what to improve next" in the
  research doc.
- **AVX-512 / NEON ADC lookup** — orthogonal to the loss choice; would
  benefit all three trainers equally.
- **Integration with `ruvector-adaptive-ann`** — pluggable trainer for
  the entropy-adaptive quantization path.
