# ADR-353: Centroid-Ball Triangle Pruning for IVF Probe Scheduling

- **Status**: Proposed (nightly research PoC)
- **Date**: 2026-10-14
- **Deciders**: nightly research agent
- **Related**: ADR-351 (edge multitenant), prior nightly ADRs on IVF variants
- **PoC crate**: `crates/ruvector-ivf-ball-pruning/`

## Context

Inverted-file (IVF) search over a learned coarse quantizer picks the top `nprobe` clusters by `d(q, c_i)` and scans every member of those clusters. Prior ruvector nightly work has explored per-vector early abandonment (`ruvector-partial-l2`), per-cluster `ef`-style scan budgets (`ruvector-ef-ivf-posting`), and norm-bucketing (`ruvector-norm-bucket-ivf`). None of these decide *whether to open a scheduled cluster at all* once the current top-k threshold is tight enough to make it unreachable.

In classical metric-tree and SA-Tree literature, triangle-inequality ball pruning is standard. We port the same idea to IVF: precompute `r_i = max_{x ∈ cluster_i} ||x - c_i||` per cluster, then at query time the lower bound on any member's distance is

```
lb(q, i) = max(0, d(q, c_i) - r_i)
```

If `lb(q, i)^2 ≥ τ²` where `τ` is the current top-k worst distance, the entire posting list can be skipped.

## Decision

Add an optional, feature-gated `BallPruneScan` to the IVF scan API. Cluster `r_max` is cached alongside posting lists (one `f32` per cluster — negligible memory). Probe ordering by `d(q, c_i)` is unchanged; the only addition is a per-probe scalar compare against the running top-k threshold.

The scan is **recall-safe**: when `nprobe` equals the number of scheduled clusters, the ball-prune output top-k is bitwise-identical to the un-pruned baseline (verified by test `ball_prune_matches_baseline`). Pruning only removes clusters that provably cannot improve the top-k.

## Consequences

**Pros**
- Zero recall loss relative to IVF baseline at the same `nprobe`.
- One-time O(N) precomputation of `r_max`, incrementally maintainable on insert/delete.
- Stackable with partial-L2 early abandonment and PQ-ADC refinement — the triangle bound reads only `d(q, c_i)` which the probe-ranking step already computed.
- Biggest wins at high recall operating points (large `nprobe`, tight `τ`): on the PoC's GMM-128d/100K dataset the ball bound discards a growing fraction of scheduled probes as top-k fills.

**Cons / failure modes**
- On uniform random data with no cluster structure, `r_max ≈ d(q, c_i)` and `lb ≈ 0`: no pruning happens, with a measurable (small) overhead from the compare. The PoC's first bench on uniform data showed this behavior — see Results.
- Insert-heavy workloads must update `r_max` monotonically; delete requires either lazy recomputation or a tombstone and periodic rebuild.
- Benefit depends on the top-k fill rate: with `k=1` and `nprobe=1`, no pruning possible.

## Alternatives considered

1. **Covering ball by percentile** (`r_p` for p=0.95 instead of `r_max`): tighter bound, but recall-lossy. Rejected for v1 — a lossy variant is listed as future work.
2. **Hierarchical ball tree over centroids**: more pruning but requires a second index structure. Overkill at IVF scales (≤ 10⁴ centroids).
3. **Convex-hull / minimum-enclosing-ball per cluster**: tighter than max-radius but harder to maintain under updates. Not justified at this stage.

## Implementation

See `crates/ruvector-ivf-ball-pruning/`:
- `IvfIndex::build` precomputes `r_max` per cluster.
- Three `IvfScan` impls: `BaselineScan`, `PartialAbandonScan`, `BallPruneScan`.
- Tests verify equivalence at `nprobe = all` and strict reduction in vectors touched under GMM data.

## Production plan

A production integration lives in `ruvector-core`'s IVF index as a feature flag `ivf-ball-prune` behind the existing scan trait. The field `r_max: Vec<f32>` is serialized next to the posting-list header. Insert path updates `r_max[ci] = r_max[ci].max(||x - c_ci||)` on each new member; delete triggers a per-cluster recompute lazily when a tombstone budget is exceeded.
