<!-- One decision, stated in the filename (ADR-0021). -->

# ADR-0001: Anisotropic score-aware product quantization for MIPS

> Decision date: 2026-09-03
> Status: Proposed
> Scope: workspace-wide MIPS codebook training (`ruvector-pq-search`, `ruvector-rabitq`, `ruvector-turboquant`, `ruvector-maxsim`, DiskANN residual codes)
> Drivers: MIPS recall at fixed code size; zero read-path change

## Context

Every workspace crate that stores compressed vectors for maximum-inner-
product search (MIPS) currently trains its quantizer under an isotropic
Euclidean-reconstruction objective inherited from Jégou et al., *Product
Quantization for Nearest Neighbor Search* (IEEE TPAMI 2011). That
objective minimises `||x - x̃||²` uniformly across every direction of
the residual `r = x̃ - x`.

At retrieval time the retriever never sees `x̃` — it sees the score
`<q, x̃> = <q, x> + <q, r>`. The distortion term `<q, r>` is dominated
by the component of `r` along the query direction. Because embedding
databases have already been trained so that each `x` scores highly with
the queries it is meant to match, the datapoint's own direction
`x̂ = x / ||x||` is a good proxy for the direction from which queries
will arrive. Errors along `x̂` hurt ranking scores far more than errors
orthogonal to it.

Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector
Quantization* (ICML 2020, arXiv:1908.10396), made this precise with the
score-aware loss

```
L_η(x, x̃) = η · ||r_∥||²  +  ||r_⊥||²        (r_∥ = (r·x̂) x̂)
```

`η = 1` recovers PQ; `η > 1` biases the codebook toward preserving the
parallel component of every residual. This is the loss powering Google
ScaNN. Nothing in `crates/` isolates it as a primitive —
`ruvector-soar-ivf` addresses a related but distinct partitioning
problem.

## Decision

1. Ship a minimal PoC crate `ruvector-anisotropic-pq` behind a
   swappable `Quantizer` trait, with two implementations: `BaselinePq`
   (η = 1) and `AnisotropicPq { eta }` (η > 1).
2. Anisotropic centroid update per subspace solves the closed-form
   normal equations `(Σ_i M_i) c = Σ_i M_i x_i`,
   `M_i = I + (η-1) û_i û_iᵀ`, via Gauss elimination with partial
   pivoting. `sub_dim` is small (8 in the default config), so this is
   `O(sub_dim³)` per centroid update — negligible next to the
   assignment step.
3. Keep ordinary L2 nearest-centroid assignment. Score-aware assignment
   differs from L2 only near cluster boundaries; L2 assignment with
   anisotropic centroid updates captures most of the gain while keeping
   the trainer a drop-in for standard PQ.
4. Keep the search path unchanged: one `M × K` inner-product LUT per
   query, sum `M` byte lookups per candidate. On-disk code layout is
   bit-for-bit compatible with any PQ index that consumes
   `Codebook::centroids` today.
5. Ship a `cargo run` benchmark that trains and evaluates three
   variants (η ∈ {1, 4, 16}) on a deterministic Gaussian-mixture
   dataset and prints MIPS Recall@10, reconstruction MSE, and timings.

## Alternatives Considered

- **Optimised Product Quantization (OPQ)** — learns a rotation matrix so
  subspaces are independent. Composes with anisotropy (different
  problem), so is a parallel improvement, not a substitute.
- **RaBitQ** (`crates/ruvector-rabitq`) — fixed-length binary sketches
  with theoretical error bounds. Different code family and memory
  budget; both belong in the workspace.
- **Anisotropy inside existing crates (`pq-search`, `rabitq`, DiskANN
  residual) directly** — the right long-term home. This ADR ships the
  isolated primitive first so the loss can be validated independently
  before being threaded through several crates.

## Consequences

**Positive.** Real, measurable MIPS-recall lift at identical code size
and query cost. Benchmark on n = 5 000, dim = 64, M = 8, K = 256:

| η | Recall@10 | MSE | Δ Recall vs baseline |
|---|-----------|-----|----------------------|
| 1 | 0.3368 | 0.0532 | — |
| 4 | 0.3590 | 0.0621 | +2.22 pp |
| 16 | 0.3680 | 0.0775 | +3.12 pp |

Zero read-path change. Trainer is ~200 lines of arithmetic plus a Gauss
solver with no external dependencies, so it compiles to WASM.

**Negative.** Reconstruction MSE gets worse — and it should: the whole
point is to trade MSE for score fidelity. Any downstream component that
assumes low reconstruction MSE (e.g., a distance-based re-ranker over
the same codes) will see degraded distances even though ranking
improves. Anisotropic training is ~1.5× slower than isotropic because
the centroid update solves a linear system instead of averaging (0.33 s
→ 0.53 s on the benchmark). Not significant for offline index builds.

**Neutral.** η is a hyperparameter, surfaced as a constructor field.
Real ScaNN uses score-aware assignment in addition to score-aware
centroid updates — remaining headroom this PoC does not exploit.

## Testable Criteria

| ID | Criterion | How verified |
|----|-----------|--------------|
| TC-1 | Anisotropic PQ (η = 4) delivers ≥ +1 percentage-point MIPS Recall@10 lift over baseline PQ at identical `(dim, M, K)` on the benchmark dataset. | Run `cargo run --release -p ruvector-anisotropic-pq --bin benchmark`; last measured value: baseline 0.3368, η=4 0.3590 (+2.22 pp). |

## References

- Guo, Sun, Simcha, Krichene, Kumar. *Accelerating Large-Scale Inference
  with Anisotropic Vector Quantization*. ICML 2020. arXiv:1908.10396.
- Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor
  Search*. IEEE TPAMI 33(1), 2011.
- Ge, He, Ke, Sun. *Optimized Product Quantization*. CVPR 2013.
- Sun, Guo, Chi, Simcha, Kumar. *SOAR: New Algorithms for Even Faster
  Vector Search with ScaNN*. NeurIPS 2023.
- ScaNN implementation:
  https://github.com/google-research/google-research/tree/master/scann
- `docs/research/nightly/2026-09-03-anisotropic-score-aware-pq/README.md`
  — this ADR's companion research doc.
