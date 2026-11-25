# ADR-361: Cross-Polytope LSH as an IVF Probe-Ordering Rule

- **Status**: Proposed (2026-11-25) — probe-ordering variant Rejected at
  current scale; kept as a research reference and as the base for
  follow-up work at higher `(d, K)`.
- **Date**: 2026-11-25
- **Related**:
  - prior ruvector sh-ivf nightly (2026-10-30), ADR-356
  - prior ruvector norm-bucket-ivf nightly
  - ADR-205 (region-pruned IVF vs plain IVF nprobe)
  - ADR-206 (PQ-IVFADC within-list pruning vs plain IVF nprobe)

## Context

The IVF probe-ordering step is the inner loop that turns a query into a
sorted list of coarse centroids to visit. For a cosine / unit-sphere
corpus, the usual rule is to rank by `-||q - c||_2^2` (equivalent to
`<q, c>` after normalization). That is `K * d` multiply-adds per query.

Cross-polytope LSH (Terasawa & Tanaka 2007; Andoni, Indyk, Laarhoven,
Razenshteyn & Schmidt NeurIPS 2015) is theory-optimal for angular LSH
and admits an `O(d log d)` pseudo-rotation per hash. In principle, with
`L` hashes we can replace the `K * d` dot-product scan by
`L * d log d + K * L` operations. For `(d, K, L) = (128, 1024, 16)` the
projected scoring speedup is ~4.3x, which is attractive.

We implemented the rule, built a real benchmark harness, and measured.

## Decision

**Reject** cross-polytope LSH collision counting as a default IVF
probe-ordering rule at ruvector's current operating scale
(`d ≤ 128`, `K ≤ 1024`). Keep the standalone crate
`crates/ruvector-cpolytope-lsh-ivf` as a reference implementation so
that follow-up research at higher `(d, K)` can start from a tested,
seeded, deterministic baseline instead of re-deriving the hash.

The L2 / cosine probe-ordering rule stays the default in ruvector-ivf.

## Consequences

**Positive**

- Honest, reproducible measurement of a seductive theoretical win. The
  nightly closes the open question "should we swap the IVF probe
  ordering for LSH on cosine data?" with a dated, seeded benchmark.
- The crate ships a working cross-polytope LSH hasher with
  pseudo-rotation. Any future ruvector feature that wants LSH (HNSW
  seeding, dual-index rerank, multi-probe shortlister) can import it.
- Clear roadmap: ADR identifies the three follow-up configurations
  (wide `d`, large `K`, SIMD WHT) where the technique may win.

**Negative**

- No immediate ruvector-ivf API change. A reader expecting the probe
  layer to gain a new backend will not find one.
- The crate adds 3 files and a `Cargo.lock` (via the detached workspace
  opt-out). It is small, but non-zero maintenance.

**Neutral**

- Cross-polytope LSH remains a known baseline. If a future vendor
  benchmark cites FALCONN-style probe ordering, we can rerun our
  harness rather than argue on paper.

## Alternatives

1. **L2 centroid ordering (default)** — `K * d` cost, exact. Winner on
   our bench at every `nprobe`. Kept.
2. **Cosine centroid ordering** — mathematically identical to L2 on
   unit-sphere centroids. Confirmed by `l2_and_cosine_agree_on_unit_sphere_order`.
3. **Spherical hashing** (Heo et al., CVPR 2012) — explored as the
   earlier `sh-ivf` nightly in this repo; also did not Pareto-dominate
   plain probe ordering at ruvector's scale.
4. **Normalized-norm bucketing** — the `norm-bucket-ivf` nightly. Lives
   in an orthogonal design dimension (residual norms, not angles) so is
   not a direct substitute.

## Benchmark summary

```text
N=50000  dim=128  K=1024  Q=1000  top_k=10  L_hashes=16
| ranker         | avg rank µs | recall@1 | recall@4 | recall@16 | recall@64 |
|----------------|------------:|---------:|---------:|----------:|----------:|
| L2             |       44.69 |   0.0326 |   0.0641 |    0.1364 |    0.3067 |
| Cosine         |       41.70 |   0.0326 |   0.0641 |    0.1364 |    0.3067 |
| CrossPolytope  |       53.61 |   0.0073 |   0.0263 |    0.0739 |    0.1653 |
```

Full research write-up:
`docs/research/nightly/2026-11-25-cross-polytope-lsh-ivf-probe/README.md`.
