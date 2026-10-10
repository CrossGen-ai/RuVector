# ADR-362: PRVQ cascade rerank for IVF-ADC pipelines

**Status:** Proposed
**Date:** 2026-10-10
**Supersedes:** none
**Related:** ADR-OPQ-rotation, ADR-pq-prefix-pruning, ADR-dyn-rerank-budget

## Context

Our IVF-ADC path (PQ-8 with `m=8`) reaches recall@10 ≈ 0.98 by shortlisting
the top-400 PQ ADC candidates and verifying each with a full-precision L2
dot. The verify step dominates on disk-resident indexes because each
candidate is a page I/O. We need the same recall with fewer full fetches.

Published composite-quantization methods (Additive Quantization,
Composite Quantization, RaBitQ) all pay for a stronger distance
estimator up front to shrink the final verify shortlist. None are
currently in ruvector; most need a non-trivial trainer. The simplest
member of the family — **PRVQ**, PQ with Residual-Vector-Quantization
on the per-vector residual — is cheap to train, strictly dominates PQ
ADC in distance accuracy, and plugs onto our existing PQ-8 index with
no changes to IVF partitioning.

## Decision

Introduce `ruvector-rvq-cascade` as a standalone crate implementing the
PRVQ cascade:

1. Train PQ-8 on IVF-local or global data (existing path).
2. Compute per-vector PQ residual.
3. Train an RVQ with `K ∈ {2, 4, 8}` whole-vector 8-bit codebooks on
   the residual.
4. Precompute per vector: `‖r_rvq‖² + 2⟨x_pq, r_rvq⟩` and partial
   recon norms.
5. At query time: build PQ L2 LUT + K RVQ dot LUTs. Compose:
   `l2sq_est = ADC_pq + ‖r_rvq‖² + 2⟨x_pq, r_rvq⟩ − 2⟨q, r_rvq⟩`.
6. Verify only the top-`N` by composite estimate.

Memory overhead vs plain PQ-8: `K` bytes/vec + 8 scalar bytes/vec (cross
term + reconstruction norm). For `K=4` and `d=128`, that's +12
bytes/vec on top of PQ-8's 8 bytes/vec — still 95% smaller than raw
float32.

## Consequences

### Positive
- Measured **2× fewer full-precision verifies** at recall@10 = 0.978 on
  N=20 000, d=128 Gaussian-mixture data.
- Backward-compatible: the PQ path is untouched; PRVQ is a new column
  on each vector record.
- Trainer is scalar, deterministic, portable; no SIMD dependency to
  reproduce the published numbers.

### Negative
- Training time: PRVQ (m=8, K=4) is ~6× slower than PQ-8 training alone
  (10.8s vs 1.8s for N=20 000). For N=10M this would be minutes vs
  seconds. Mitigation: train PRVQ per IVF cell in parallel.
- Memory per vector: +12 B/vec at `K=4`. At N=10M that is +120 MB.
- Online latency in RAM: +25% ms/query vs classic PQ-rerank at same
  recall. **The win is in the fetch dimension**, not pure RAM latency.

### Alternatives considered
- **AQ / Composite Quantization**: better recall-vs-bytes frontier but
  trainer is non-trivial (joint optimization). Deferred.
- **RaBitQ (`ruvector-rabitq` exists)**: strong theoretical bounds but
  binary-only; complementary rather than competing with PRVQ. Future
  work: combine as PRVQ for coarse, RaBitQ for finer refinement.
- **Just increase the PQ shortlist**: free recall but linear in full
  fetches, which is what we want to avoid.

### Risks
- Over-fitting PRVQ to training data when IVF cells are small
  (< 2 000 vectors). Mitigation: fall back to global PRVQ for small
  cells, per-cell for large.
- The cached cross term is bit-identical to a scalar per vector; it
  must survive serialization. Add to the index snapshot format.

## Rollout

- Phase 1 (this ADR): standalone crate, bench, docs. **No workspace
  wiring**, same convention as recent nightly research crates (whss,
  dyn-rerank-budget).
- Phase 2 (future ADR): optional feature in `ruvector-ivf-core`, flag
  `--features prvq`, per-cell training path.
- Phase 3 (future ADR): FastScan AVX2/NEON SIMD backend for the online
  LUT scan.

## Decision

Accepted at prototype tier. Promote to production tier after a Phase 2
ADR with disk-resident benchmark numbers from a 10M-vector corpus
(SIFT1M or Deep1M as the staple).
