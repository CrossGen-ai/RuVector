# ADR-352: PQ Prefix Pruning — Partial-Sum Early Abandon for 4-bit FastScan ADC

- **Status**: Proposed (nightly research, 2026-10-01)
- **Author**: nightly research agent
- **Related**: ADR on PQ FastScan (lut16), ADR on partial-L2 early abandon, prior
  nightly `pq-fastscan-lut16-2026` and `partial-l2-early-abandon-2026`

## Context

RuVector's PQ FastScan path (`ruvector-pq-fastscan-lut16`) accumulates all M
per-segment distances for every PQ code before deciding whether the code enters
the top-k. For a corpus of N codes and M segments, cost is Θ(N·M) nibble
look-ups and additions. On real top-k queries with k ≪ N, the vast majority of
codes are far from the query — once their partial sum exceeds the running
k-th-best distance (`pivot`), they cannot win.

FastScan's single-pass accumulator never checks `pivot` mid-code, because the
natural L2/scalar FastScan loop is organised segment-outer / code-inner for LUT
locality. Prior work has considered per-code abandon, but it breaks SIMD
parallelism.

## Decision

Introduce **prefix-pruning scanners** that:

1. Operate on fixed-width SIMD-friendly groups of `LANE = 32` codes.
2. Maintain a per-lane running sum and an `alive` mask.
3. Track a **suffix floor** = sum of per-segment minima for yet-unvisited
   segments.
4. After each segment, mark a lane dead when `partial_sum + floor_remaining ≥
   pivot`. Pruning is lossless because `floor_remaining` is a true lower bound
   on the yet-to-add distance.
5. Visit segments in **descending LUT variance** (`VarOrderedPrune`): segments
   where the query sees a wider spread of centroid distances separate codes
   faster, raising the pivot pressure early and killing more lanes with fewer
   lookups.

The implementation lives in a standalone crate `ruvector-pq-prefix-pruning`
with three scanners sharing a `Scanner` trait: `Baseline`, `PrefixPrune`,
`VarOrderedPrune`. It is drop-in alongside the existing FastScan crate.

## Consequences

### Positive
- 2.1–2.7× end-to-end speedup over FastScan baseline at k = 10 on 100k and
  500k code corpora, with M ∈ {16, 32} and d = 128.
- Pruning is **lossless** — produces identical top-k ids and distances as
  baseline (verified in `prune_matches_baseline_on_random_data`).
- Zero recall change: no quantization or approximation on top of PQ itself.
- Variance ordering adds ~O(M log M) sort cost per query, trivially amortised.
- Group-of-32 lane layout is friendly to future AVX2 (32×u8) / NEON (16×u8 ×
  pair) SIMD lowering.

### Negative
- At **large k** (k = 100 on 100k codes), pivot stays loose and pruning
  overhead (branch per lane, mask maintenance) can be a *net loss* — ~5% to
  ~15% slower than baseline in our measurements. The scanner choice must be
  gated on expected k vs N ratio.
- Group-level abandon is coarser than per-code: a single surviving lane keeps
  all 32 lanes alive. Workloads with highly correlated clusters (where codes
  within one group often tie) may show less pruning than per-code abandon
  would.
- Adds a tiny per-query cost: variance computation and suffix-floor array
  (`O(M)`), negligible for M ≤ 64 but worth noting.

### Neutral
- The ADC score space is unchanged. Downstream rerank, IVF probing, and HNSW
  neighbour inspection are untouched.

## Alternatives considered

1. **Per-code (not per-group) abandon.** Tighter pruning but breaks SIMD lane
   parallelism; typically slower than Baseline on modern CPUs for M ≤ 32.
2. **Pure variance ordering, no pruning.** Does not reduce lookups; wastes the
   information. Rejected.
3. **Dynamic re-ordering per group** (reorder segments based on partial-sum
   derivative). More compute per group; empirically did not beat a per-query
   fixed order on our workloads.
4. **Scale-aware u16 saturation.** We quantise the LUT to u8 with a single
   per-query scale. A per-segment scale would tighten `seg_min` but requires a
   per-segment back-conversion for the pivot comparison — not worth the
   complexity for M ≤ 32.

## Rollout

- Keep as a standalone crate; promote to feature-gated scanner inside
  `ruvector-pq-fastscan-lut16` after one more round of validation on real
  (SIFT-1M / Deep-10M-slice) workloads.
- Expose `PrefixPrune` as the default when `k / N ≤ 1e-3`, `Baseline`
  otherwise. The decision is cheap and static.
- Add `scanner_stats` metric to `ruvector-metrics` so operators can see
  per-query prune rate in production.
