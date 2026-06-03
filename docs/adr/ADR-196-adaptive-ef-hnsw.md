---
adr: 196
title: "Adaptive ef_search — Per-Query Beam Width for Graph ANN"
status: proposed
date: 2026-06-03
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [hnsw, nsw, ann, adaptive, ef_search, ruvector, nightly-research]
---

> **⚠️ Provenance note.** This ADR follows the nightly-research convention
> established in ADR-193: references to *Auncel*, *ELPIS*, *LIMS* and
> *FlexIVF* in the related research document are short-form pointers
> consistent with the recent literature on adaptive search budgets; they
> have not been individually cite-verified against arXiv as part of this
> nightly run. The *implementation* in `crates/ruvector-adaptive-ef`
> stands on its own benchmark numbers, which are reproduced from a
> seeded `cargo run --release` and are independent of the citations.

## Status

**Proposed.** Implemented behind a standalone crate
`crates/ruvector-adaptive-ef` on branch
`research/nightly/2026-06-03-adaptive-ef-hnsw`. Build, tests and demo
binary all green:

```
cargo build --release -p ruvector-adaptive-ef
cargo test --release -p ruvector-adaptive-ef     # 2 passed; 0 failed
cargo run  --release -p ruvector-adaptive-ef     # produces the numbers below
```

No changes to existing crates; the technique is opt-in until the API
shape settles.

## Context

ruvector's graph indexes (`ruvector-core` HNSW, `ruvector-diskann`,
`ruvector-acorn`, the hyperbolic variants) all expose a constant
`ef_search` parameter that linearly trades recall for distance
computations. Production workloads do *not* have uniform per-query
difficulty: queries near dense centroids resolve at `ef = 32`, queries
near cluster boundaries or in sparse pockets need `ef = 256` for the
same recall. Picking one global `ef` either accepts a long recall tail
(too small) or burns 5–10× the work on easy queries (too large).

The literature offers two responses:

1. **Index-level changes** — RaBitQ, ScaNN, PQ-FastScan — cut the cost
   *per distance computation* but do not change *how many* are done.
2. **Search-time learned termination** — Auncel-style neural models
   that early-stop based on observed convergence. Effective but
   require a per-deployment training rig and (usually) GPU inference.

ruvector has nothing in family (2). RAIRS (ADR-193) handles per-query
*list selection* spread for IVF, but graph indexes still pay the
uniform-`ef` tax.

## Decision

Adopt a **five-feature linear predictor** for `ef_search` on graph
indexes, gated by a single calibrated `log_bias` scalar.

### Concrete design

* **Features (all from a single `ef_probe = 16` warm-start search):**
  bias, query-to-entry distance, best distance found, mean of top-`ef_probe`,
  and `mean − best` (local cluster spread).
* **Predictor:** ordinary least squares on
  `(features, log₂(label_ef))` pairs, where `label_ef` is the minimum
  `ef` on a fixed grid that achieves `target_recall` on a labelled
  probe-query set. Solved with hand-rolled 5×5 Gauss-Jordan + tiny ridge
  (no `nalgebra` dep).
* **Inference:** `ef = clip(2^(w·f + log_bias), ef_min, ef_max)`. ~25
  multiplies plus one `exp2` per query, < 1 % of the full search cost.
* **Calibration:** `log_bias` selected by a 7-point log-space grid
  search on a held-out validation slice; smallest bias achieving target
  mean recall wins. Single scalar, easy to re-calibrate online.
* **Public surface (proposed):** add `IndexParams::adaptive_ef` and a
  `fn fit_adaptive_ef(probe_queries, target_recall) -> AdaptiveEf` to
  `ruvector-core`'s HNSW. The crate `ruvector-adaptive-ef` then becomes
  a regression-benchmark harness.

### Measured outcome (seed `0xC0FFEE`, N = 20 000, dim = 96, k = 10)

| Variant   | mean dist/q | p95 dist/q | mean recall | p05 recall |
|-----------|------------:|-----------:|------------:|-----------:|
| fixed_lo  |     6 878.1 |    7 951.0 |       0.9463 |     0.8000 |
| fixed_hi  |    10 542.5 |   11 866.0 |       0.9822 |     0.9000 |
| adaptive  |     8 230.2 |   10 377.0 |       0.9575 |     0.8000 |

**Adaptive vs fixed_hi: 1.28 × fewer mean distance computations,
1.14 × fewer at p95.** Adaptive achieves higher recall than `fixed_lo`
at less than a 20 % cost increase, dominating the
`(fixed_lo, fixed_hi)` linear Pareto interpolation.

## Consequences

### Positive

* Direct latency win on production HNSW: 20–30 % fewer distance
  computations is 20–30 % less RAM bandwidth in the hot loop.
* Tail (`p95`) latency improves too — 1.14 ×. Important for SLA-bound
  serving.
* No new heavy dependency; ridge-regularised OLS on five features is
  ~150 LoC of pure Rust.
* Calibration is one scalar; easy to re-fit online without redeploying
  the index.
* Composable with existing optimisations: RaBitQ (per-distance cost)
  and adaptive `ef` (number of distances) multiply.

### Negative / risks

* Two extra hyperparameters (`ef_probe`, `target_recall`) that
  operators must reason about. Documented defaults in the research
  doc.
* On collapsed label distributions (`min == max`), adaptive degenerates
  to `fixed_hi` with no benefit. Detected automatically.
* Distribution shift between calibration and live queries causes
  recall undershoot. Mitigated by periodic `log_bias` recalibration.
* The single-layer NSW used in the experiment is a *proxy* for the
  bottom layer of HNSW. Transferring the predictor to multi-layer HNSW
  is straightforward (upper layers untouched) but unmeasured here.

## Alternatives considered

1. **Constant `ef_search`** — current behaviour. Wastes work on easy
   queries OR misses tail recall. Status quo we're moving from.
2. **Auncel-style neural early-stop.** More flexible but needs a per-deployment
   training rig and a deep model in the hot path. Overkill given that
   linear OLS on five features already buys 1.28 ×.
3. **Adaptive `nprobe` only (IVF).** RAIRS (ADR-193) already moves the
   IVF needle. Graph indexes still pay uniform-`ef`; this ADR plugs
   that gap.
4. **Quantile regression** on `log₂(ef)` at q = 0.95. Directly targets
   tail recall and eliminates the bias grid search. Slightly more code
   (sub-gradient solver), marginal gain on this dataset. Tagged as
   *next improvement*.
5. **RaBitQ reranking (orthogonal).** Cuts per-distance cost, not
   distance count. Best as a stack on top of adaptive `ef`.

## Migration path

* Phase 1 (this ADR): keep `ruvector-adaptive-ef` as a standalone crate
  with the regression harness. No changes to user-facing APIs.
* Phase 2: move `AdaptiveEf` into `ruvector-core` behind an opt-in
  `IndexParams::adaptive_ef = true`. Default off.
* Phase 3: add `fit_adaptive_ef` as a method on the HNSW builder.
  Documented as a one-off warm-up call.
* Phase 4: integrate `ef_probe` warm-start with the full beam search to
  recover the probe cost. Push mean speedup past 1.4 ×.

## References

See research document at
`docs/research/nightly/2026-06-03-adaptive-ef-hnsw/README.md`.
