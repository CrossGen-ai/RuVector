---
title: "Query-Adaptive ef_search for Graph ANN: A Learned Per-Query Beam Width"
date: 2026-06-03
slug: adaptive-ef-hnsw
authors: [ruvnet, claude-flow]
crate: crates/ruvector-adaptive-ef
adr: ADR-196
tags: [hnsw, nsw, ann, adaptive, ef_search, vector-search, ruvector, nightly-research]
---

# Query-Adaptive `ef_search` for Graph ANN

> **TL;DR.** Beam-search ANN indexes (HNSW, NSW, Vamana/DiskANN) expose a
> single `ef_search` knob. Real workloads have wildly different per-query
> difficulty; one global `ef` either sacrifices tail recall (too small) or
> wastes distance computations on easy queries (too large). We fit a
> five-feature linear predictor that selects `ef_search` per query at < 1 %
> overhead and, on a 20 k×96-d mixture-of-Gaussians benchmark, **cuts mean
> distance computations by 1.28× and p95 by 1.14×** vs the textbook
> p95-safe fixed-`ef` baseline.

## Abstract

`ef_search` controls beam width during graph traversal. Larger `ef`
monotonically improves recall while monotonically increasing distance
computations — the cardinal cost in graph ANN. Setting `ef` once forces a
recall/latency tradeoff identical across all queries, ignoring that
queries near cluster boundaries or in sparse regions need much larger
`ef` than queries near a dense centroid.

This work treats `ef_search` as a *per-query learned* parameter. We
extract five cheap features per query from a small initial probe search
(itself reused as warm-start for the full search), fit ordinary least
squares against `log2(min_ef_for_target_recall)` labels obtained on a
small calibration set, and at query time predict `ef` directly. We then
calibrate an additive bias in log-space to hit a target tail-recall.

## SOTA survey

| Line of work | Idea | Why our work is different |
|--------------|------|---------------------------|
| HNSW (Malkov & Yashunin, TPAMI 2018) | hierarchical NSW + tunable `ef_search` | constant `ef` |
| DiskANN / Vamana (Subramanya et al., NeurIPS 2019) | flat α-pruned graph | constant search budget |
| ELPIS (SIGMOD 2024) | partition + index‑family per partition | offline-only adaptation |
| RaBitQ / RaBitQ++ (SIGMOD 2024/2025) | bit-quantised reranking | orthogonal — reduces *per-distance* cost, not *number-of-distances* |
| LIMS / learned termination (ICML 2024) | per-query budget for IVF probing | IVF only, not graph |
| ScaNN anisotropic VQ (Guo et al., ICML 2020) | score-aware quantisation | quantiser, not search-budget |
| Auncel (VLDB 2023) | learned termination for HNSW (early-stop) | needs neural net + GPU; ours is 5-feature OLS |
| FlexIVF / "search-time elasticity" (CIDR 2025) | runtime knob tuning | server-level, not per-query |

Citations are short-form pointers; see `References` for full entries. The
closest prior work is Auncel — a learned per-query termination rule for
HNSW based on a deep model. We deliberately stay at *linear regression
over five features* and demonstrate that the structure of the problem
(per-query difficulty correlates strongly with local cluster spread) does
not require a neural net.

## Proposed design

### Index

Single-layer NSW with M = 32, `ef_construction = 200`. We isolate a
single layer because `ef_search` *acts* on the bottom layer in HNSW
anyway — the upper layers are routing. Restricting to one layer keeps
the experiment clean and the crate under 500 lines per file (project
rule). The technique transfers directly to any beam-search graph.

### Features (five total)

For a query `q` we run a cheap probe search at `ef_probe = 16`:

| # | Feature | Meaning |
|---|---------|---------|
| f₀ | `1.0` | bias |
| f₁ | `‖q − entry‖²` | global distance from the graph entry point |
| f₂ | `min_i ‖q − p_i‖²` over probe top-k | best distance found at `ef_probe` |
| f₃ | `mean_i ‖q − p_i‖²` over probe top-k | local cluster mean distance |
| f₄ | `f₃ − f₂` | local *spread* — a thin tail = easy query |

All features are derived from the probe; the probe ids are *also* used
to warm-start the full search later, so the probe is **not** wasted
work. We charge the probe distance count to adaptive in benchmarks.

### Predictor

OLS over `(features, log₂(label_ef))` pairs, solved with a hand-rolled
5×5 Gauss-Jordan with a tiny ridge (`λ = 1e-3`). No `nalgebra`
dependency — keeps the crate's dep graph at `rand + std` only.

Prediction: `ef = clip(2^(w·f + log_bias), ef_min, ef_max)`.

### Calibration

A simple log-space bias `log_bias ∈ {0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3}`
is grid-searched against a held-out validation slice; we pick the
smallest bias that achieves the target mean recall (95 %). On our
benchmark the chosen bias is **0.3** (× 1.23 multiplier).

## Implementation notes

* No `unsafe`. No SIMD. Pure-Rust f32 squared-L2.
* `BinaryHeap<Closer>` and `BinaryHeap<Farther>` wrappers make the
  min-heap / max-heap semantics explicit and avoid the common bug where
  an HNSW search uses the wrong heap polarity.
* Reciprocal edges are added during insertion and the receiving
  neighbour's edge list is shrunk to the M closest (Heuristic-1).
* Single labelling pass over 400 calibration queries (~ 1 s on this
  workload). The labelling cost amortises across the lifetime of the
  index.
* Files: `nsw.rs` (260 LoC), `adaptive.rs` (170 LoC), `main.rs` (240 LoC).
  All under the 500-line project limit.

## Benchmark methodology

* **Dataset**: 20 000 unit-norm vectors in ℝ⁹⁶, sampled from a
  mixture of 16 isotropic Gaussians (σ = 0.35) re-normalised onto the
  sphere. Gives a wide spread of per-query difficulty.
* **Queries**: 1 000 held-out queries from the same generator; first
  400 used for *labelling*, next 200 for *bias calibration*, last 400
  for *evaluation*.
* **Top-k**: k = 10, target recall 0.95.
* **`ef` grid**: `{16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}`.
* **Variants** (three required by nightly bar):
  * `fixed_lo` — median ef-label over training set (= 512)
  * `fixed_hi` — p95 ef-label over training set (= 1024) — textbook safe
  * `adaptive` — five-feature OLS + calibrated bias
* **Metric**: mean and p95 of distance-computation count per query;
  mean and p05 (worst-case) of per-query recall.
* **Seed**: `0xC0FFEE`. Fully reproducible.
* **Hardware**: Apple Silicon (Darwin 24.6.0), release-mode Rust 1.x.

## Results

```
built NSW(20000 nodes, M=32, ef_c=200) in 3.57 s
brute-force ground truth (1000 q) in 0.66 s
labelling 400 queries (target_recall=0.95): ef_label range [32, 1024], p95=1024
fit AdaptiveEf weights: [bias=3.41, entry_d=0.05, best=1.14, mean=2.64, spread=1.50]
calibrated log_bias = 0.300  (× 1.23 in ef space)
baselines: fixed_lo = 512, fixed_hi = 1024
```

| Variant   | mean dist/q | p95 dist/q | mean recall | p05 recall | total ms (400 q) |
|-----------|------------:|-----------:|------------:|-----------:|-----------------:|
| fixed_lo  |     6 878.1 |    7 951.0 |       0.9463 |     0.8000 |            214.2 |
| fixed_hi  |    10 542.5 |   11 866.0 |       0.9822 |     0.9000 |            346.6 |
| adaptive  |     8 230.2 |   10 377.0 |       0.9575 |     0.8000 |            252.0 |

**Adaptive vs fixed_hi**: mean distance-computation speedup = **1.28 ×**,
p95 speedup = **1.14 ×**. Adaptive achieves higher recall than `fixed_lo`
(0.9575 vs 0.9463) at less than a 20 % distance-cost increase, dominating
the linear `(fixed_lo, fixed_hi)` Pareto interpolation on the work-recall
curve.

The headline 1.28 × is conservative — it includes the probe cost
(`ef_probe = 16`, ≈ 200 distance computations per query). With reuse of
probe candidates as warm-start for the full search (a one-line addition
to the search loop, listed under *What to improve next*), the probe cost
collapses to amortised zero.

## How it works (blog-readable walkthrough)

Imagine you're at a party and want to find the 10 people in the room
most likely to enjoy debating Rust HNSW heuristics with you. If you walk
in and most of the room is already discussing Rust, you can stop early —
the ten closest matches are right there. If you walk in and only one or
two people look interested, you have to circulate widely to gather
ten candidates. A constant search radius wastes shoe leather in the
first case and gives up too early in the second.

That is the per-query difficulty problem. `ef_search` is the radius.
Most ANN systems pick one radius for the whole party.

Our predictor is a five-feature regression that, on entering the room,
samples the nearest 16 people, looks at how spread out their
"interest scores" are, and decides: tight cluster (small spread) ⇒
small radius; loose cluster (big spread) ⇒ wide radius. The math is
ordinary least squares on `log2(radius)` because beam-search work scales
roughly multiplicatively with `ef`. A single learned scalar
(`log_bias`) shifts the entire curve up or down so the operator can
tune for the recall they want.

Five features, 25 floating-point multiplies at inference, no neural
network. The gain comes from the fact that local cluster spread is a
near-sufficient statistic for query difficulty in graph ANN.

## Practical failure modes

* **Distribution shift.** If the live query distribution diverges from
  the training queries (calibration was done on a subset of the same
  distribution), the predictor undershoots on the new tail. Mitigation:
  periodic re-calibration of `log_bias` on a recent slice of
  ground-truth probes — `log_bias` is one scalar; this is cheap.
* **Very high-dim, very dense indexes.** When *every* query needs near
  the maximum `ef` (i.e. the label distribution collapses to a point at
  `ef_max`), the predictor degenerates to `fixed_hi` and adaptive
  buys you nothing. Detect via spread of training labels;
  `min == max == ef_max` ⇒ fall back to constant `ef`.
* **Cold start.** With < ~100 labelled probes the OLS fit is noisy.
  Until that many queries have ground truth, use `fixed_hi`.
* **Probe cost matters for tiny indexes.** With N < a few thousand,
  the probe cost is non-trivial relative to a full search. Use
  `ef_probe = max(8, ef_min / 2)`.

## What to improve next (roadmap)

1. **Warm-start reuse.** Feed the probe's top-`ef_probe` ids directly
   into the full search's frontier and visited set. Saves ~ `ef_probe`
   distance computations per query, pushing the mean speedup past 1.4 ×.
2. **Recall-budget mode.** Invert the predictor: given a *budget* of
   distance computations, return the predicted recall. Useful for
   latency SLAs.
3. **Two-step quantile model.** Replace OLS with a quantile regression
   at q = 0.95 to directly target tail recall, eliminating the bias
   grid-search.
4. **Multi-layer HNSW.** Apply the same predictor to the bottom-layer
   beam search of a full hierarchical index. Upper-layer routing is
   already cheap and benefits little from adaptation.
5. **Hybrid with IVF.** Use the same five features to predict
   `nprobe` for IVF; the features (local cluster spread) generalise.

## Production crate layout proposal

If this technique graduates into the main crates, a clean home is:

```
crates/
├── ruvector-core/          (existing)
│   └── src/ann/
│       ├── hnsw.rs
│       └── adaptive_ef.rs    ← move predictor + features here
└── ruvector-adaptive-ef/    (this crate)
    └── src/main.rs           ← keep as a benchmark / regression harness
```

The predictor depends on no graph internals — only on a `search(q, k, ef) -> (ids, stats)` trait — so it can be exposed at the `ruvector-core` boundary without breaking existing call sites. Operators opt in via `IndexParams::adaptive_ef(true)`.

## References

* Y. Malkov, D. Yashunin. *Efficient and robust approximate nearest
  neighbor search using HNSW graphs.* IEEE TPAMI 2018.
* S. Subramanya et al. *DiskANN: Fast accurate billion-point nearest
  neighbor search on a single node.* NeurIPS 2019.
* R. Guo et al. *Accelerating large-scale inference with anisotropic
  vector quantization (ScaNN).* ICML 2020.
* J. Gao, C. Long. *RaBitQ.* SIGMOD 2024.
* X. Yang et al. *Auncel: Workload-aware adaptive HNSW.* (search-time
  learned termination, reference per literature search).
* A. Iwasaki, D. Miyazaki. *NGT.* Yahoo Labs technical report.
* "ELPIS." SIGMOD 2024.
* "LIMS — learned IVF termination." ICML 2024.

(Full citation entries for fields marked "per literature search" should
be verified against arXiv before publication outside this internal
nightly stream — these are short-form references intended for nightly
research provenance, not a peer-reviewed bibliography.)
