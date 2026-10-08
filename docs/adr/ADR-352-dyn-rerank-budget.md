# ADR-352: Dynamic Rerank-Budget Policy for IVF-PQ

**Status:** Proposed
**Date:** 2026-10-08
**Branch:** `research/nightly/2026-10-08-dyn-rerank-budget`
**Related:** ADR-345 (mincut-gated forgetting), ruvector-pq-prefix-pruning,
ruvector-cascade-adc.

## Context

Every rerank-based ANN pipeline in ruvector today — IVF-PQ, SPANN,
PQ-prefix-pruning, cascade-ADC — reranks the top `rerank_k`
approximate candidates with full-precision L2. The constant is picked
at index-build time, baked into the config, and never revisited. We
measured on 20K × 128d data that **Fixed(200) spends 2× the rerank work
of Fixed(100) for a 2.9 pp recall gain**, which means most of the
compute is paid for easy queries that would have hit the same top-k at
R = 50.

A simple per-query predictor on the coarse-scan output can reclaim that
work. We want one that:

* has **no retraining** when the corpus drifts (ruvbrain, edge
  tenants, agent memory all drift continuously),
* costs **O(1) extra per query**,
* ships as a **drop-in trait** so Fixed stays the default while DRB
  rolls out behind a feature flag.

## Decision

Introduce a `BudgetPolicy` trait in a new standalone crate
`ruvector-dyn-rerank-budget`:

```rust
pub trait BudgetPolicy {
    fn choose(&self, cands: &[Cand], topk: usize) -> usize;
    fn name(&self) -> &'static str;
}
```

Ship two implementations:

* `Fixed(usize)` — the current behavior, remains the default.
* `DrbGap { r_min, r_max, probe, gap_ref }` — opt-in. Reads one
  pair of approximate scores at indices `k − 1` and `k + Δ`,
  divides a calibrated reference gap by the observed gap, clamps,
  and returns R.

Calibration is one offline pass over a cached query set that
produces a single `f32` (`gap_ref`) stored next to the index. The
oracle variant (`Oracle`) is included as a measurement tool only;
it is not shippable.

Measured gain on 20K × 128d synthetic Gaussian: DRB-Gap(15, 150, Δ=32)
reaches recall@10 = 0.562 at mean R = 143.1, vs Fixed(200) at
recall 0.573 / mean R = 200 — a 28 % rerank-cost cut for 1.1 pp
recall loss. The policy is per-query, so operators tune `(r_min,
r_max)` the way they tune `nprobe` today.

## Consequences

**Positive**

* Rerank-bound deployments cut compute without a retrain cycle.
* Trait boundary isolates future policies (streaming `gap_ref`,
  multi-probe gap, learned MLP) without touching the IVF or PQ paths.
* Fixed stays the default — the ADR carries no behavioral change
  for existing consumers.

**Negative**

* One extra `f32` per index to store `gap_ref`.
* Operators now have two knobs (`r_min`, `r_max`) instead of one
  (`rerank_k`). Documentation + defaults needed.
* On uniform-noise coarse scorers (e.g. aggressive 1-bit RaBitQ)
  the policy collapses to `Fixed(r_max)`. Need to warn operators
  via the calibrator (if `std(gap_q) / mean(gap_q)` < threshold,
  log a hint that DRB will not help).

**Neutral**

* The policy is deterministic given `(cands, k, gap_ref)`. No new
  non-determinism in the retrieval path.

## Alternatives considered

1. **Learned MLP budget predictor** (AdaptNN-style). Rejected for
   nightly research: ties DRB to a training loop and adds a
   dependency on an embedding-model-specific feature extractor,
   which breaks the "no retraining when corpus drifts" guarantee.
2. **Early-stop during rerank** (DiskANN-style tapering). Rejected
   as *insufficient*, not wrong — early-stop reads the exact
   distances to decide, so it still pays for the first N rerank
   calls. DRB cuts that N before any exact distance is computed.
   The two compose: DRB sets R; early-stop can still prune within R.
3. **Variance of full approx vector.** Rejected: dominated by the
   long tail of far candidates; carries little signal about the
   local uncertainty around position k.
4. **Per-query learned nprobe (AdaptNN).** Addresses a different
   bottleneck (coarse-stage recall ceiling). Compatible with DRB,
   not a substitute.
5. **Static `rerank_k` tuned per-cluster.** Captures corpus-level
   variance but ignores query-level variance, which is where the
   remaining compute lives.

## Rollout

* Phase 1 (this ADR): land the standalone crate on the research
  branch with tests + bench. Measured numbers in
  `docs/research/nightly/2026-10-08-dyn-rerank-budget/README.md`.
* Phase 2 (next): wire `BudgetPolicy` into `ruvector-pq-search` and
  `ruvector-spann` behind a `drb` feature flag. Default stays
  `Fixed(rerank_k)`.
* Phase 3: add streaming `gap_ref` estimator so recalibration is
  free. Publish a comparison on a real embedding corpus
  (`sift1m`, `glove-100`) before promoting DRB to default.
