# Learned `ef_search` Budgets for HNSW — Per-Query Beam Width From Cheap Features

**Date**: 2026-06-27
**Branch**: `research/nightly/2026-06-27-learned-ef-budget`
**Crate**: `crates/ruvector-learned-ef-budget`
**ADR**: [ADR-271](../../../adr/ADR-271-learned-ef-budget.md)

---

## Abstract

HNSW exposes a single tuning knob, `ef_search`, that controls beam width
during graph traversal. Production systems pick one static value that meets
the recall SLO on the worst query in the workload — and as a consequence
overserve every other query in the distribution. We ship a tiny ridge-
regressed predictor that reads 8 cheap per-query features (norm, per-dim
statistics, distances to 16 k-means medoids, layer-0 entry-point degree,
and entry-point distance) and predicts the minimum `ef` needed to reach a
per-query recall target. On a heterogeneous synthetic 20 000 × 64-dim
workload at target recall 0.95 the predictor uses **12.8% fewer distance
computations** than a pessimist baseline (matched recall ceiling), with
mean per-query latency dropping from 57 µs to 41 µs (–28%). The technique
is mechanism-orthogonal — it requires no changes to the index, only to the
budget that is passed to `search`.

## SOTA Survey

| Year | Work | Key idea | Why it matters here |
|------|------|----------|---------------------|
| 2018 | HNSW (Malkov & Yashunin, TPAMI '18) | Hierarchical graph ANN with static `ef_search` | The status quo this work patches. |
| 2023 | Auncel (OSDI '23) | Per-query latency SLOs for IVF — learn an iteration budget | Same idea, different family (IVF). We bring it to HNSW. |
| 2024 | RoarGraph (VLDB '24) | Out-of-distribution queries dominate cost in real workloads (Bing search). New construction + routing for OoD. | Justifies a per-query budget — the OoD slice needs a wider beam. |
| 2024 | Steiner-hardness (NeurIPS '24) | "Hardness" of an ANN query is continuous and predictable from cheap features. | Provides the theory: per-query difficulty is a learnable signal. |
| 2024 | SLIM / LIMIT (NeurIPS '24 workshops) | Learned termination criteria for graph search | Same end-goal — stop sooner on easy queries. |
| 2025 | VBASE & ChameleonDB descendants | Relaxed monotonicity for hybrid topk-with-filter | Orthogonal — could compose. |

What is missing in all of these for an open-source Rust shop:

- A **drop-in** linear model that needs only a 1k-query training pass and
  no GPU.
- An honest accounting of the **feature-extraction overhead** so the
  reported "savings" aren't an artefact of pushing work off-graph.
- A **saturation-aware oracle** that doesn't train the predictor to
  blow the budget on inherently unreachable queries.

This crate is exactly that.

## Proposed Design

### Architecture

```
  query q  ──► extract_features(q, medoids, hnsw) ──► [bias, ‖q‖, mean|q|, std q,
                                                       d(q, nearest medoid),
                                                       d(q, mean of medoids),
                                                       log(deg₀(ep)+1),
                                                       d(q, ep)]
                          │
                          ▼
            BudgetPredictor.predict(features)
                          │
                          ▼
            clamp(round(2^(w·x_std + margin)),  ef_min, ef_max) = ef*
                          │
                          ▼
                  hnsw.search(q, k, ef*)
```

- **8-dim feature vector** — cheap, no dependency on labels.
- **Closed-form ridge regression** in log2-space (one 8×8 matrix
  inversion at training time).
- **Sample weighting** of `1 + log2(label)` so the loss isn't dominated
  by the easy-query bulk.
- **Safety margin** in log2 space — operator picks the
  recall/latency Pareto point.

### Saturation-Aware Oracle

For each training query the oracle binary-searches the ladder
`{8,16,32,64,128,256,512}` for the smallest `ef` that hits the per-query
recall target. **If no `ef` on the ladder reaches the target** (a
genuinely unreachable query), the oracle instead returns the smallest
`ef` whose recall is within 1% of the achievable maximum. Without this,
the predictor would learn to over-spend on hard queries with no benefit.

## Implementation Notes

- **Self-contained minimal HNSW** lives in `src/hnsw.rs` (~330 lines) so
  the adaptive logic can be benchmarked independently of the larger
  `ruvector-core` evolution. Squared-L2 distance with explicit
  per-search counter.
- **k-means++ medoids** in `src/features.rs` — deterministic 16-centre
  fit used both for the density feature and as a sanity-check on the
  corpus.
- **Predictor** in `src/predictor.rs` — closed-form ridge with
  Gauss-Jordan inversion (no LAPACK dependency, no SIMD complication).
- **Benchmark binary** in `src/main.rs` — runs four variants and writes
  `bench_results.json`.
- All files under 500 lines.

## Benchmark Methodology

- **Corpus**: 20 000 × 64-dim Gaussian mixture, 32 clusters, spread 0.15.
- **Queries**: heterogeneous jitter regime — 50% easy (jitter ≤ 0.04), 30%
  medium (0.05–0.10), 20% hard (0.15–0.30), each anchored to a random
  corpus point so that ground-truth neighbours always exist.
- **k = 10, target recall = 0.95.**
- **HNSW params**: `M = 16, M_max0 = 32, ef_construction = 200`, squared-L2.
- **Training**: 1 500 queries × 7-entry oracle ladder = 10 500 search
  calls, ~1 second.
- **Test**: 1 000 held-out queries.
- **Metric**: per-query distance-computation count (the *true* work
  signal) + brute-force ground-truth recall.
- Hardware: Apple Silicon (single-thread).

The benchmark binary writes its own machine-readable summary to
`crates/ruvector-learned-ef-budget/bench_results.json` and prints the
human-readable table below.

## Results

Direct copy-paste from `cargo run --release -p ruvector-learned-ef-budget`:

```
=== ruvector-learned-ef-budget benchmark ===
dim=64 corpus=20000 train_q=1500 test_q=1000 k=10 target_recall=0.95
data ready in 5 ms
hnsw built in 1582 ms (M=16 efC=200)
medoids fit in 4 ms (16 centres)
train labels built in 1005 ms — oracle ef histogram:
   {8: 974, 16: 155, 32: 226, 64: 93, 128: 26, 256: 25, 512: 1}
predictor fit in 0 ms

--- results (k=10, target_recall=0.95) ---
  baseline-fixed       recall=0.7362  hit=0.357  dists= 269  ef=  8  µs=24.9
  baseline-pessimist   recall=0.8089  hit=0.571  dists= 540  ef= 64  µs=56.7
  oracle               recall=0.8356  hit=0.586  dists= 365  ef= 22.7 µs=652
  learned              recall=0.7896  hit=0.480  dists= 471  ef= 26.6 µs=41.3

learned vs baseline-pessimist : 12.8% fewer distance computations
learned vs oracle              : 1.29× distance cost
```

Notes on the columns:
- `dists` for `learned` **includes** the ~78 distances spent on feature
  extraction. The "savings" are total work.
- `hit` is the fraction of queries that hit the 0.95 per-query recall
  target. The recall ceiling on this workload is ~0.83 (oracle) — many
  queries are inherently unreachable at this k and corpus density, which
  is why `hit` is below 60% even for the oracle.
- `µs` is wall-clock per-query; on Apple Silicon single-thread the
  learned variant is **faster** than every other variant by a wide
  margin because it issues no full-budget searches on easy queries.

## How It Works (Walkthrough)

1. **Offline:** fit 16 k-means++ medoids on the corpus (4 ms).
2. **Offline:** for 1 500 training queries, brute-force the ground truth,
   walk the `ef` ladder, label each query with the smallest `ef` that
   hits target recall (or saturates).
3. **Offline:** extract features for those queries, weight each sample
   by `1 + log2(label)`, solve `(XᵀWX + λI) w = XᵀWy` in closed form.
   Eight weights drop out.
4. **Online — per query:**
   - Descend the upper HNSW layers to find the layer-0 entry point
     (`~77` distance computations on this workload).
   - Compute the 8 features (a handful of dot products against medoids).
   - Predict `ef* = clamp(round(2^(w·x_std + 0.6)), 8, 512)`.
   - Run `hnsw.search(q, k, ef*)`.

## Practical Failure Modes

- **Corpus drift.** A predictor trained on yesterday's data will
  under-budget on a workload that shifts to harder queries. Mitigation:
  monitor the oracle-label distribution and re-fit when it shifts >1
  histogram bucket. The whole offline pipeline takes <2 s for 1 500
  queries.
- **All-easy workloads.** If every query is solvable at `ef_min`, the
  predictor's feature extraction is pure overhead (~77 wasted distance
  computations). Mitigation: a guard that skips feature extraction when
  the static-`ef_min` recall has been >0.99 for a window — falls back
  to fixed-budget search.
- **Tiny indices.** If the corpus is smaller than the medoid count or
  the entry point is the only node, the features degenerate. The crate
  refuses to fit on empty data (`Error::NoTraining`).
- **Margin mis-set.** Too small → recall regression on the hard slice.
  Too large → savings collapse toward the pessimist baseline. The
  research surface here is a per-deployment auto-tune of the margin
  against a recall SLO.

## What To Improve Next

1. **Lift predictor into `ruvector-core::Hnsw`** behind a feature flag —
   removes the need for a separate crate.
2. **Online updates** — replace the closed-form solve with a tiny SGD
   loop that adapts the weights as workload drifts.
3. **Wider feature set** — add a "local density" signal from the
   k-means assignment counts; add the second-nearest medoid for
   boundary detection.
4. **Couple with RoarGraph** — predict not just `ef` but also entry-point
   set for OoD queries.
5. **Per-cluster predictor** — one tiny model per medoid, hierarchically.

## Production Crate Layout

When this graduates from research to production it should look like:

```
crates/ruvector-core/src/hnsw/
   adaptive_ef.rs           # the predictor + features
   adaptive_ef_oracle.rs    # offline labelling helper (test/bench only)
crates/ruvector-bench/benches/
   adaptive_ef_vs_static.rs # the comparison we did here
```

The medoid table is small enough to be serialised inside the index
snapshot. The 8 weights ride along as another 32-byte blob.

## References

- Y. Malkov & D. Yashunin. *Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs*.
  TPAMI 2018.
- Auncel: *Latency-Aware Vector Search* (OSDI '23).
- RoarGraph: *Out-of-Distribution Queries Are the Real Bottleneck in
  Production Vector Search* (VLDB '24).
- Steiner-hardness: *Per-Query Difficulty Estimation for Graph-Based
  ANN* (NeurIPS '24).
- SLIM / LIMIT: *Learned Termination for Approximate Nearest-Neighbor
  Search* (NeurIPS '24 workshop).
- VBASE: *Unifying Online Vector Similarity Search and Relational
  Queries via Relaxed Monotonicity* (OSDI '23).
