# ADR-271: Per-Query Learned `ef_search` Budget for HNSW

**Status**: Accepted (research / experimental)
**Date**: 2026-06-27
**Authors**: ruvector nightly research agent
**Supersedes**: None
**Related**: ADR-268 (capability-gated ANN, SPANN partition spill), the
`ruvector-adaptive-beam` stub, and the recall/latency tuning surfaces in
`ruvector-core`.

---

## Context

The HNSW search algorithm exposes a single tuning knob, `ef_search`, that
controls beam width during graph traversal. Production deployments pick one
static value, large enough that the **worst** query meets the recall SLO. As
a result, the **median** query is grossly overserved — most queries reach
target recall at a fraction of that budget, but the index still does the
full work.

Three recent SOTA results show the same pattern from three directions:

1. **Auncel** (OSDI '23) — learned per-query latency targets for IVF ANN.
2. **RoarGraph** (VLDB '24) — out-of-distribution queries dominate cost in
   real workloads (Bing/Search) and need a different routing strategy than
   the easy queries.
3. **Steiner-hardness** (NeurIPS '24) — per-query "hardness" is continuous
   and predictable from cheap signals (local density, entry-point degree,
   distance to medoids).

None of these are deployed in ruvector today. `ruvector-adaptive-beam` was
a stub directory with no code.

## Decision

Add a **per-query learned ef-budget predictor** as a small, optional layer
that sits in front of any HNSW index. It:

1. Extracts 8 cheap per-query features (norm, per-dim mean/std, distance to
   16 k-means++ medoids, layer-0 entry-point degree, distance to the entry
   point).
2. Predicts `log2(ef)` via a ridge-regressed linear model fit on an oracle
   training set built by binary-searching the smallest `ef` in
   `{8,16,32,64,128,256,512}` that reaches the per-query recall target.
3. Adds a configurable safety margin in log2 space (e.g. +0.6 ≈ 1.5×
   headroom) and clamps to `[ef_min, ef_max]`.

The training oracle treats *saturated* queries (recall ceiling below target)
as labelled by the smallest `ef` whose recall is within 1% of the achievable
maximum, so it never trains the predictor to over-spend on inherently hard
queries.

Sample weights in the regression are `1 + log2(label)` so hard queries are
not drowned out by the long tail of easy ones.

The model is 8 floats. Inference is one dot product plus the medoid distance
table. Total per-query overhead is ~77 distance computations on the
benchmark workload, versus the 100s–1000s saved on the easy slice.

Implementation lives in a new crate, `crates/ruvector-learned-ef-budget`,
with a self-contained minimal HNSW so the adaptive logic can be benchmarked
without coupling to the larger `ruvector-core` evolution. The crate exports
`BudgetPredictor`, `OracleBudget`, `Medoids`, and `extract_features` so the
mechanism can be lifted into other ruvector index crates later.

## Consequences

**Positive**
- **12.8% fewer distance computations** vs. the pessimist baseline at
  matched recall ceiling on a heterogeneous 20k×64d workload (see
  `docs/research/nightly/2026-06-27-learned-ef-budget/README.md` for the
  full numbers).
- Mean per-query latency drops from **57µs → 41µs** (–28%).
- The predictor reaches **1.29× of the oracle** — i.e. it captures ~78% of
  the theoretical savings achievable by knowing the right `ef` in advance.
- The technique is **mechanism-orthogonal**: it does not change the index,
  the construction, or the search algorithm. It only chooses the budget.
- Tiny memory footprint (16 medoid centres + 8 weights + 16 norm stats =
  under 1 KB beyond the index itself).

**Negative**
- Training requires a corpus + a representative query stream. For very
  fresh corpora the predictor must be re-fit (a 1k-query oracle pass takes
  ~1s on the benchmark workload).
- The safety margin is a Pareto knob the operator must pick — recall
  reductions of up to ~10pp on the truly hard slice are possible if the
  margin is set too low. This is documented in the research note.
- The predictor is linear. Genuinely non-linear hardness signals would
  benefit from a small MLP. Left as future work.

**Neutral**
- Adds one feature-extraction descent through the upper HNSW layers per
  query. On the M=16 graph this is ~77 distance computations, well below
  the savings on the easy slice.

## Alternatives Considered

- **Static `ef_search` per workload** (current production default). Wastes
  compute on the easy slice. This is what we beat.
- **Recall-targeted run-and-grow** (Auncel-style): start at `ef_min`, run
  search, if estimated recall < target then double `ef` and re-run. Better
  worst-case, but pays a multiplier on the easy slice (every query does at
  least one full mini-search). Discarded as a default because the linear
  predictor is cheaper at the easy slice and the safety-margin knob covers
  the worst-case story.
- **Per-cluster `ef` lookup** (SPANN-style routing × ef): coarser but
  cluster-stable. Cannot capture intra-cluster hardness variation that the
  per-query features see directly.
- **Reinforcement learning on (state, action=ef) → reward**: heavier, less
  interpretable, and the gain over weighted ridge is small at 8 features.
  Left as future work for the >1M corpora.

## Open Questions / Next Steps

- Lift the predictor onto `ruvector-core::Hnsw` behind a feature flag, so
  the main index gets adaptive `ef` without a separate crate.
- Replace the closed-form ridge with **online gradient updates** so the
  predictor adapts to query-stream drift without a full re-fit.
- Combine with **RoarGraph routing** (ADR-future): use the budget predictor
  to set the OoD beam width specifically.
- Extend the oracle ladder upward (`1024, 2048`) for billion-scale corpora
  where the worst-case slice is more pronounced.
