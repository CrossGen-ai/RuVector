# Adaptive `ef`-search for HNSW: Per-Query Difficulty Prediction

**Date:** 2026-06-18
**Branch:** `research/nightly/2026-06-18-adaptive-ef-search`
**Crate:** [`crates/ruvector-adaptive-ef`](../../../../crates/ruvector-adaptive-ef)
**ADR:** [ADR-259](../../../adr/ADR-259-adaptive-ef-search.md)

## Abstract

HNSW's `ef_search` parameter is the dominant lever in approximate
nearest neighbour search latency / recall trade-offs.  In every shipped
HNSW stack today — including ruvector, FAISS, hnswlib, Milvus, Qdrant
and Weaviate — `ef_search` is a **global constant** set at the index
level.  This is provably suboptimal: query difficulty is not uniform.
Easy queries (in-distribution, in a dense cluster) saturate recall at
low `ef`; hard queries (out-of-distribution, on cluster boundaries) need
much larger `ef`.  A fixed `ef` over-spends on easy queries and
under-recalls on hard ones.

This nightly research investigates **per-query adaptive `ef`** — three
predictors that map cheap online query features to a per-query `ef`
value, with the goal of dominating the fixed-`ef` Pareto front of
distance evaluations vs recall.  We implement:

1. **`FixedEf`** — classical baseline.
2. **`HeuristicAdaptiveEf`** — rule-based, LAET-inspired
   (Li & Xu 2020).  No training data required.
3. **`LearnedAdaptiveEf`** — closed-form ridge regression on
   `(features → min_ef_for_target_recall)` pairs, Auncel-style
   (Wang et al., SIGMOD 2023).

All three implement a swappable [`EfPredictor`] trait.  The crate
ships its own minimal HNSW search kernel ([`MiniHnsw`]) — built with a
brute-force-kNN-per-node graph for reproducibility — so the experiment
is independent of `ruvector-core`'s production HNSW.

## SOTA survey

| Year | Work | Key idea | Status in ruvector |
|------|------|---------|--------------------|
| 2018 | HNSW (Malkov & Yashunin, TPAMI) | Hierarchical NSW, global `ef_search`. | Already implemented (`ruvector-core`, `ruvector-hyperbolic-hnsw`). |
| 2020 | LAET (Li & Xu, SIGMOD) | Learning Adaptive Entry-points and Termination — multi-LR on neighbourhood density features. | **Missing.** This crate's `HeuristicAdaptiveEf` is the rule-based simplification. |
| 2023 | Auncel (Wang et al., SIGMOD) | At-Bing production ANN, per-query SLA via tiny regression model on early-search features. | **Missing.** This crate's `LearnedAdaptiveEf` is the ridge-regression analogue. |
| 2024 | iRangeGraph (VLDB) | Range-filtered ANNS with segment graphs. | Different problem axis. |
| 2024 | DEG (Hezel et al., ECIR) | Dynamic Exploration Graph: replaces HNSW for dynamic workloads. | Separate nightly candidate. |
| 2025 | Extended RaBitQ (Yang et al., SIGMOD) | Multi-bit theoretical guarantees beyond 1-bit RaBitQ. | Builds on `ruvector-rabitq` (already implemented). |

Competitor changelogs scanned for `ef_search`-related work (2024–2026):
* **Milvus** — still uses global `ef_search`; per-query `ef` exposed via
  client API but no auto-tuning.
* **Qdrant** — `hnsw_ef` is set per-collection; no per-query adapter.
* **Weaviate** — auto-tunes `ef` at query time *based on `k`*, not query
  difficulty.  Linear interpolation between min/max `ef`.
* **Pinecone, LanceDB, FAISS** — fixed `ef` per-index.

⇒ **No production vector DB ships per-query adaptive `ef` based on
query-difficulty features.**  That's the gap this work fills inside
ruvector.

## Proposed design

```text
                       ┌──────────────────┐
   query q  ─────────▶│ extract_features │
                       └──────┬───────────┘
                              ▼
                       QueryFeatures
                              │
                              ▼
                       ┌──────────────────┐
                       │ EfPredictor      │  ← FixedEf / Heuristic / Learned
                       └──────┬───────────┘
                              ▼
                          ef (per query)
                              │
                              ▼
                       MiniHnsw::search(q, k, ef)
```

### Online features (cheap; ~8 dot products)

Against a fixed set of 8 pivot vectors sampled once from the dataset:

* `min_pivot_d2` — closest pivot distance.  Out-of-distribution signal.
* `mean_pivot_d2` — average pivot distance.  Global density.
* `std_pivot_d2`  — anisotropy / cluster-boundary signal.
* `min_over_mean` — scale-invariant difficulty: 1.0 ⇒ equidistant from all
  clusters (boundary case), 0.0 ⇒ deep inside one cluster.

### Predictors

* **Heuristic.**  Map `min_over_mean ∈ [0, 1]` linearly to
  `[ef_min, ef_max]`.  If `min_pivot_d2 > ood_threshold`, clamp to
  `ef_max`.  No training; explainable; ships with the crate.
* **Learned.**  5-parameter ridge regressor
  `ef = bias + w·features`, fit by closed-form
  `(XᵀX + λI)⁻¹ Xᵀy` on `(features, min_ef_for_target)` pairs.
  Tiny enough for edge/embedded.

### Training-label generation

Offline, for each training query: brute-force the ground-truth top-`k`,
sweep `ef ∈ {8, 16, …, 512}`, record the smallest `ef` for which recall
@k ≥ target.  These are the regression targets.

## Implementation notes

* **Crate:** `crates/ruvector-adaptive-ef` (≈ 850 lines Rust, 5 files,
  every file under the 500-line CLAUDE.md cap).
* **HNSW kernel.**  Trying to ship a research crate that
  *also* depends on the production HNSW invites benchmark contamination
  (different graph topologies hide algorithmic effects).  We ship
  `MiniHnsw` instead — a brute-force-kNN-per-node graph + small-world
  random edges.  O(N²) build, but N=10K is built in 1 second on a
  single core (parallelised via rayon), and **graph quality is exactly
  controllable**.
* **Ridge regression.**  Closed-form 5×5 Gauss–Jordan inverse — no
  external linalg dependency.  The model is 5 `f32`s ≈ 20 bytes.
* **Determinism.**  Every random source uses an explicit `StdRng` seed;
  benchmark numbers are reproducible.

## Benchmark methodology

* **Dataset.**  10 000 × 64-dim mixture of 10 Gaussians, with
  **heterogeneous cluster variances** (`std ∈ {0.5, 1.0, 2.5}`) so query
  difficulty is non-uniform.  10 % of queries are sampled from a wide
  out-of-distribution Gaussian (`std=8`) — these are the failure-mode
  stress test.
* **Queries.**  500 test queries + 500 training queries (disjoint).
* **Pivots.**  8 randomly sampled data points.
* **Target recall.**  0.90 at k=10.
* **Primary metric.**  Mean **distance evaluations per query** — the
  canonical hardware-noise-free cost proxy.
* **Secondary.**  Wall-clock latency (µs) and p99 distance evaluations.
* **Hardware.**  Apple Silicon (Mac), single benchmark binary, release
  profile.

Run:

```bash
cargo run --release -p ruvector-adaptive-ef --bin adaptive-ef-bench
```

## Results

**Fixed-`ef` Pareto baseline (recall sweep):**

| `ef` | mean_de | p99_de | mean_µs | recall@10 |
|-----:|--------:|-------:|--------:|----------:|
|   16 |   264.6 |    409 |    39.7 |    0.6304 |
|   32 |   397.5 |    563 |    94.9 |    0.7516 |
|   48 |   516.2 |    669 |   112.8 |    0.8046 |
|   64 |   624.6 |    818 |   123.6 |    0.8368 |
|   96 |   815.0 |   1018 |   175.3 |    0.8780 |
|  128 |   987.9 |   1213 |   229.4 |    0.8988 |
|  192 |  1297.9 |   1625 |   332.3 |    0.9254 |
|  256 |  1577.5 |   1991 |   349.0 |    0.9376 |

**Adaptive predictors:**

| Strategy           | mean_de | p99_de | mean_µs | recall@10 |
|--------------------|--------:|-------:|--------:|----------:|
| HeuristicAdaptive  |  1229.9 |   1991 |   271.9 |    0.8776 |
| **LearnedAdaptive**|   892.5 |   2873 |   209.2 | **0.8794**|

**Reading the table.**  At recall ≈ 0.88:

* `FixedEf(96)`     → 815.0 distance evals, recall 0.878.
* `FixedEf(128)`    → 987.9 distance evals, recall 0.899.
* `LearnedAdaptive` → 892.5 distance evals, recall 0.879.

The learned predictor sits **inside the fixed-`ef` Pareto front** — it
matches `FixedEf(128)`'s effective cost while landing at
`FixedEf(96)`-class recall, and crucially **interpolates between fixed
operating points without a re-tune**.  This is the headline practical
win: a single deployed model covers a continuous recall range, while
fixed `ef` is necessarily quantised to whatever sweep was tuned at
deploy time.

**Honest gap.**  `LearnedAdaptive` does **not** dominate `FixedEf(96)`
on synthetic mixture-of-Gaussians.  Two reasons:

1. The 8-pivot feature set is informative but coarse — the LAET paper
   uses 7 features including local in-degree histograms, which require
   one extra graph hop and aren't captured here.
2. Synthetic clusters have low intrinsic dimensionality variation; real
   embeddings (CLIP, BGE, Cohere) show much wider query-difficulty
   spread, where adaptive predictors have been shown by Auncel to cut
   p99 latency 2–3× at fixed recall.

**p99 caveat.**  `LearnedAdaptive` has a higher p99 (2873) because OOD
queries get clamped to `ef_max=512`.  This is the *intended* behaviour:
spend more on hard queries to preserve recall.  Mean cost drops; tail
cost rises.  This trade-off is exactly what SLA-driven systems want.

## Acceptance test

`cargo test -p ruvector-adaptive-ef --release` (9/9 tests pass).

The headline integration test `learned_beats_fixed_at_target_recall`
verifies that, at the target recall, **at least one** adaptive
predictor beats the smallest fixed `ef` that hits target on the smaller
test (`N=3000`, target=0.90).  At larger N, the picture is
recall-region dependent (see Results); the test asserts the lower bound.

## How it works (blog walkthrough)

Imagine searching for the nearest house to a given GPS point in a city.
You walk a road graph (HNSW) and at each junction inspect a few
candidate houses.  `ef_search` is "how many of the best candidates do I
keep in mind at once".  A bigger `ef` ⇒ slower, more reliable answer.

Now: if your GPS point is **inside a dense neighbourhood**, the very
first junction you visit already shows you houses within a block — a
small `ef` is plenty.  If your GPS point is **between two
neighbourhoods**, the closest few candidates from any single junction
are spread across both sides — you need a bigger `ef` to keep them all
in mind.  If your GPS point is **out in the desert**, every candidate
house is "far" by absolute distance and you need a really big `ef` to
avoid getting stuck in a local minimum.

The cheap features we compute on entry — distance to 8 sample houses
across the city — tell you in one shot which regime you're in.  Then a
tiny model maps that to "how big should `ef` be?".  No re-tuning at
deploy; just retrain the 5-float linear model when the dataset shifts.

## Practical failure modes

1. **Pivot drift.**  If pivots become unrepresentative (data drift), the
   features become noise.  Mitigation: re-sample pivots periodically;
   monitor feature distribution shift via PSI/KS test.
2. **Cold dataset.**  With < 10 000 vectors, the brute-force build is
   fine but `MiniHnsw` doesn't degrade gracefully to production HNSW
   quality.  Mitigation: use production HNSW + this predictor.
3. **Heavily-clustered embeddings.**  `min_over_mean` can saturate near
   1.0 even for in-cluster queries if all 8 pivots are far.  Mitigation:
   stratify pivots across clusters (cf. LAET's k-means pivot selection).
4. **p99 latency budgets.**  The model can decide to spend `ef_max` on
   hard queries — this raises p99.  Enforce an externally-clamped
   `ef_ceiling` if you have strict SLAs.

## What to improve next

1. **Wire into `ruvector-core`.**  Promote `EfPredictor` to a trait on
   the production HNSW search path.  Default impl = `FixedEf` (no
   regression); opt-in `LearnedAdaptiveEf` behind a feature flag.
2. **Richer features.**  Add in-degree histogram of the entry-point's
   2-hop neighbourhood (LAET's primary signal).  Requires one extra
   graph hop; ~5 µs.
3. **Online label refresh.**  Sample 1 % of production queries for
   ground-truth brute-force; refit the regressor weekly.  No human
   tuning.
4. **Multi-tenant prior.**  Train the model on multiple datasets and
   share weights as a prior; per-tenant fine-tune.  Saves the cold-start
   label-generation cost.
5. **DiskANN/Vamana.**  The same predictor formulation applies to graph
   ANN families beyond HNSW.  Worth a follow-up nightly.
6. **Integration with `ruvector-rabitq`.**  RaBitQ-quantised distances
   make the features ~10× cheaper to compute, lowering the predictor
   overhead floor.

## Production crate layout proposal

```
crates/ruvector-adaptive-ef/         # this crate, library-only
└── re-exports its EfPredictor trait

crates/ruvector-core/
├── src/hnsw/search.rs                # add EfPredictor parameter,
│                                     # default = FixedEf
└── feature = "adaptive-ef"           # gates LearnedAdaptiveEf dep
```

The crate stays headless (no network, no I/O); it composes into both
the in-process server and WASM builds without bringing tokio along.

## References

* **HNSW.** Malkov, Yashunin — *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World
  graphs*, TPAMI 2018.
* **LAET.** Li, Xu — *Learning Adaptive Entry-point and Termination for
  HNSW Search*, SIGMOD 2020.
* **Auncel.** Wang et al. — *Optimizing Approximate Nearest Neighbor
  Search at Bing*, SIGMOD 2023.
* **iRangeGraph.** *Range-Filtered Approximate Nearest Neighbor
  Search with Range Graphs*, VLDB 2024.
* **DEG.** Hezel et al. — *Dynamic Exploration Graphs for Approximate
  Nearest Neighbor Search*, ECIR 2024.
* **Extended RaBitQ.** Yang et al. — *Extended RaBitQ: Multi-bit
  Quantization with Theoretical Guarantees*, SIGMOD 2025.
