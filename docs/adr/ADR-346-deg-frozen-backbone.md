# ADR-346: DEG-Style Occlusion Graph with a Per-Node Frozen Delaunay Backbone

## Status

Experimental. Nightly PoC only (`crates/ruvector-deg-frozen-backbone`,
2026-09-19). Not promoted; conditional recommendation for future
merge into `ruvector-diskann` or `ruvector-hnsw-delta-neighbors` behind
a feature flag, contingent on layered-variant validation at ≥100k
scale.

## Context

ruvector's family of dynamic ANN indexes (`ruvector-diskann`,
`ruvector-coherence-hnsw`, `ruvector-hnsw-delta-neighbors`, etc.)
uniformly treat all graph edges as equally mutable: every insert may
prune any edge it can reach. This is fine for one-shot builds but
degrades under mixed insert/query workloads because RNG-diverse edges
established early — the ones that form the effective Delaunay skeleton
— are indistinguishable from cheap edges added late during search
warmup, and can be replaced by later, locally closer, globally
redundant candidates.

The Dynamic Exploration Graph (Neuhaus & Schubert, 2024) formalizes
the "swap-if-better" occlusion rule but demonstrates the same problem
in a harder form: pure DEG loses substantial recall against a static
HNSW build unless a background refinement thread runs.

`ruvector-hnsw-delta-neighbors` (nightly 2026-09-09) approached this
from the *observability* side — tracking which edges changed since the
last snapshot. This ADR takes the *policy* side: at insert time, mark
the strongest RNG-diverse edges as immutable ("frozen"), let the rest
follow DEG rewiring.

## Hypothesis

```text
Given a dynamic ANN index where each node stores M out-edges chosen by
Vamana-style RNG occlusion pruning against ef_build=64 candidates:

  H1: Freezing the closest ⌈α·M⌉ edges at insert time (α∈[0,1]) prevents
      later inserts from displacing high-quality Delaunay-approximation
      edges.

  H2: With α∈[0.25, 0.50], recall@10 on a clustered synthetic corpus
      (n=5000, dim=64) will be within 3 recall points of a static
      HNSW-lite baseline, whereas α=0 (pure DEG-mutable) will lose ≥20
      recall points.

  H3: Insert median wall-clock will not exceed the baseline by more
      than 10%.
```

## Decision

Ship the PoC crate `ruvector-deg-frozen-backbone` on the research
branch as evidence. **Do not** promote to production yet. The
promotion gate is:

1. Reproduce the recall-gap closure at ≥100k vectors on a real
   embedding corpus (SIFT1M or GloVe-100).
2. Add a layered variant (HNSW top layers on top of the frozen
   backbone) since the single-layer PoC is recall-limited at
   n≥20k.
3. Provide a documented `α`-tuning heuristic (adaptive by local density
   or by cumulative insert count) rather than a fixed constant.

## Consequences

### Positive

* **Recovers recall under continuous inserts.** At n=5 000, frozen(α=0.5)
  achieves 0.9535 recall@10 vs 0.7500 for pure DEG (baseline HNSW-lite
  0.9755). At n=20 000, 0.594 vs 0.466 (baseline 0.718). H1 and H2
  hold at n=5 000; H2 misses the "within 3 points" bound at n=20 000
  because the single-layer topology is the dominant recall factor at
  that scale.
* **No query-time penalty.** Frozen edges participate in search
  identically to mutable edges. QPS is within 5 % of baseline across
  variants.
* **Insert latency parity.** Median insert 24–48 µs vs baseline
  27–70 µs; H3 holds.
* **Trivially safe reads of frozen edges.** A future concurrent-insert
  implementation only needs to lock the mutable suffix.

### Negative

* **Sensitive to distribution drift.** If the embedding distribution
  shifts substantially over the index's lifetime, the frozen edges
  become stale. No drift-detection or demotion is implemented.
* **Single-layer PoC does not scale.** Recall at n=20 000 is ≤0.72 even
  for the baseline; a layered variant is needed before this is
  meaningful at production scale. This is a limitation of the PoC's
  scope, not of the frozen-backbone idea per se.
* **Delete semantics unspecified.** Deleting a node whose id is in
  another node's frozen prefix must trigger repair. `ruvector-hnsw-repair`
  is the natural home for that logic and must be wired up before
  promotion.
* **Non-adaptive α.** A constant α is the wrong long-term default —
  the results already show α should scale with n.

### Neutral

* Additional per-node memory: one `usize` (`frozen_prefix`). At 8 bytes
  per node, negligible next to the `M × 4` bytes of neighbor ids.

## Alternatives considered

1. **Do nothing — keep the fully mutable HNSW.** This is the current
   ruvector default. Works well when inserts are bulk and rare; the
   nightly hypothesis is specifically about interleaved insert/query
   workloads which are increasingly common in agent-memory scenarios.
2. **FreshDiskANN-style two-segment split.** Immutable long-term
   segment + mutable delta segment, merged periodically. This is a
   coarser granularity — per-segment rather than per-node — and forces
   a periodic merge that this ADR's design avoids.
3. **Continuous refinement (DEG-CR).** Neuhaus & Schubert's follow-up:
   a background thread walks the graph and improves edges. Orthogonal
   to and compatible with frozen-backbone; complementary work.
4. **Full static rebuild on trigger.** Simple, robust, but destroys
   incremental-insert throughput and pauses the index. Frozen-backbone
   trades that pause for a predictable, bounded recall dip.

## Follow-ups

* Bench against `ruvector-diskann` on a real 100k-scale corpus.
* Prototype adaptive α based on local intrinsic dimensionality (LID).
* Wire delete-with-frozen-invalidation through `ruvector-hnsw-repair`.
* Consider merging with `ruvector-hnsw-delta-neighbors` — its edge-change
  tracking is a natural fit for the mutable suffix.

## References

* Nightly research doc:
  `docs/research/nightly/2026-09-19-deg-frozen-backbone/README.md`
* PoC crate: `crates/ruvector-deg-frozen-backbone/`
* Related nightlies: 2026-09-09 (hnsw-delta-neighbors),
  2026-09-07 (landmark-alt-hnsw), 2026-06-18 (hnsw-delete-repair).
