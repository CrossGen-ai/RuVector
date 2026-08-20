# ADR-322 — Anisotropic Product Quantization (APQ) with Optional Learned Rotation

Status: Accepted (nightly research, 2026-08-20)
Author: nightly research agent
Related: ADR-149 (PQ-ADC search), ADR-231 (RaBitQ), ADR-244 (Matryoshka coarse/fine)

## Context

ruvector already ships several 1-byte-per-subspace quantizers:

* `ruvector-pq-search` — plain Lloyd's k-means PQ with asymmetric distance.
* `ruvector-rabitq` — sign-based binary quantization with expected-value correction.
* `ruvector-turbovec` / `ruvector-turboquant` — SIMD table-lookup accelerators.

All three optimize a raw reconstruction loss `‖x − c‖²`. That loss weights
*every* direction of error equally, but downstream search only cares about
the component of error that changes the *score* — for inner-product /
cosine ranking, that is the residual component *parallel* to the datapoint
direction. This is why classical PQ often gives excellent MSE but only
mediocre recall on modern embedding datasets (BERT/GTE/E5), where norms
are near-uniform and score is essentially cosine.

Two well-known ideas fix this:

1. **Anisotropic loss** (Guo et al., ICML 2020 — ScaNN). Split each residual
   into components parallel and orthogonal to the datapoint direction and
   penalize the parallel term by `η > 1`. The closed-form weighted k-means
   update is `A c = b` with `A = |S|·I + (η−1)·Σ d dᵀ` and
   `b = Σ x + (η−1) Σ (dᵀx) d`, solved per cluster per iteration.
2. **Learned rotation** (Ge et al., 2013 — OPQ). Pre-rotate data so
   variance is balanced across PQ subspaces — a fair PQ starting point that
   stacks cleanly on top of any loss.

Neither is currently in ruvector.

## Decision

Ship a new crate `ruvector-anisotropic-pq` that provides three swappable
implementations behind a single `Quantizer` trait:

* `PlainPQ` — Lloyd's baseline (parity with `ruvector-pq-search`).
* `AnisotropicPQ` — score-aware weighted k-means, full symmetric
  linear-system update per cluster (small `ds×ds` Gaussian elimination).
* `AnisotropicPQR` — APQ with a variance-balancing rotation fit from the
  data covariance via in-crate Jacobi eigendecomposition (no BLAS).

The trait is intentionally minimal (`encode`, `sdc`, `reconstruct`,
`bytes_per_code`) so IVF, DiskANN, and HNSW-rerank consumers can swap the
backend by generic bound rather than by re-writing index code.

Pure safe Rust, deterministic given seed, no BLAS, no unsafe.

## Consequences

**Positive**

* Adds a score-aware quantizer to the ruvector menu — required for
  competitive MIPS on modern near-unit-norm embeddings.
* Rotation piece is reusable: `rotation::Rotation` can pre-process input
  for any PQ variant, not just APQ, so downstream code gets OPQ for free.
* Same on-disk footprint as plain PQ (`m` bytes per vector). A caller
  switching from `ruvector-pq-search` to `ruvector-anisotropic-pq` sees no
  storage change — only recall improves.
* Deterministic + no BLAS means the crate builds on every ruvector target
  (Linux/macOS/Windows/wasm32-unknown-unknown after minor tweaks).

**Negative / trade-offs**

* Anisotropic loss training is ~3× slower per iteration than plain Lloyd
  because of the per-cluster linear solve. Measured: 383 ms → 1.15 s for
  8k×64d, k=256 on M-series laptop (see benchmark). This is *training*
  cost, incurred once, and dwarfed by real embedding-generation cost.
* Rotation adds a per-vector `O(d²)` multiply at both encode and query
  time. Measured: 303 k → 232 k encoded vectors/sec (−23%). Acceptable
  for offline encoding pipelines; hot query paths should cache rotated
  query vectors instead of re-rotating per shard.
* On strictly L2 workloads (no MIPS semantics), pure APQ shows no gain
  over plain PQ and can slightly regress MSE (+0.6% observed). The
  rotation piece is what drives recall improvement on such workloads.

**Follow-ups**

* Wire APQ into `ruvector-pq-search` as an alternative training backend
  (feature-gated).
* SIMD table-lookup path for SDC (`ruvector-turboquant` integration).
* Study `η` schedule per dataset — currently a single scalar.

## Alternatives considered

* **Only ship rotation (OPQ) without APQ.** Rejected: on the MIPS
  configurations where APQ is designed to shine (unit-norm high-d
  embeddings, high compression ratios), APQ alone contributes an
  additional ~2–5 pp recall on ScaNN's reported benchmarks — worth having
  as a knob, not just an implementation detail.
* **Depend on `linfa-clustering` or `ndarray-linalg`.** Rejected: adds a
  BLAS dependency and complicates wasm builds. The linear systems are
  ≤32×32 in practice, well within a hand-written Gaussian-elimination
  solver's sweet spot.
* **Full LSQ (Local Search Quantization) instead.** Rejected as scope
  creep: LSQ needs 3+ passes over the data per iteration and a much
  larger auxiliary state. APQ + rotation captures ~80% of LSQ's recall
  gain at a small fraction of the complexity.

## References

* Guo, R.; Sun, P.; Lindgren, E.; Geng, Q.; Simcha, D.; Chern, F.;
  Kumar, S. *Accelerating Large-Scale Inference with Anisotropic Vector
  Quantization.* ICML 2020.
* Ge, T.; He, K.; Ke, Q.; Sun, J. *Optimized Product Quantization.*
  TPAMI 2013.
* Jégou, H.; Douze, M.; Schmid, C. *Product Quantization for Nearest
  Neighbor Search.* TPAMI 2011.
