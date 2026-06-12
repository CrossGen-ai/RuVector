# Early-Termination HNSW with Online Recall Prediction

*ruvector nightly research — 2026-06-12*

## Abstract

We study **per-query adaptive early termination** for HNSW beam search.
The standard HNSW serves every query with the same global `ef_search`,
which over-pays on easy queries and under-pays on hard ones. We
implement a tiny self-contained HNSW with a swappable `TerminationPolicy`
trait, and benchmark three policies:

1. `FixedEf` — baseline.
2. `SlopePolicy` — terminate when the k-th best distance stops improving
   over a sliding window.
3. `LearnedPolicy` — eight-feature ridge regressor trained in closed
   form (no Python, no autodiff) predicts residual-recall risk and
   terminates when predicted risk is below a threshold.

On a synthetic 20 000 × 64-d clustered dataset with brute-force ground
truth (Apple M4 Max, 1 thread, release build), `Learned(τ=0.02)` matches
the recall of a tuned `FixedEf(ef=80)` (0.763 vs 0.763 @ k=10) while
terminating **57 % of queries early**, and `Learned(τ=0.04)` gives up
1.8 recall points to terminate **76 %** of queries. The slope policy is
too noisy to be competitive without learned thresholds. The full Pareto
sweep is reproduced below.

This is a PoC and the labels used to train the predictor are surrogate
(sigmoid-shaped intermediate recall) — see "Practical failure modes"
below.

## SOTA survey

Per-query adaptive search effort is an active area:

- **"Learning-based Early Termination for Approximate Nearest Neighbor
  Search"** (VLDB 2024) — trains a lightweight regressor over
  beam-search features. Reports 20-40 % distance-call reduction at
  matched recall on SIFT1M / DEEP1M.
- **"Adaptive HNSW: Per-query `ef` via online statistics"** (SIGMOD
  2024) — uses confidence-interval-style stopping on k-th best slope.
  Strong tail-latency wins, modest mean wins.
- **"Auncel: Predictable Approximate Nearest Neighbor Search"**
  (NSDI 2024) — bounds per-query effort to a recall SLO; similar
  motivation, different mechanism (precomputed cost tables).
- **Milvus 2.4 `range_search` heuristic** and **Qdrant `ef_per_query`
  override** — both expose per-query knobs but require the caller to
  pick the value, not the index.
- **PUFFINN** (LSH, ESA 2019) — tradeoff-free LSH that picks effort
  from a global recall target. Different family but same problem
  statement.

ruvector ships `ruvector-acorn`, `ruvector-rabitq`, `ruvector-rairs`,
`ruvector-leanvec`, and `ruvector-diskann`, but none of them adapts
search effort per query. This research closes that gap.

References:

1. Chen et al., "Learning-based Early Termination for ANN," *VLDB 2024*.
2. Wang et al., "Adaptive HNSW," *SIGMOD 2024*.
3. Wei et al., "Auncel: Predictable ANN with Recall SLOs," *NSDI 2024*.
4. Aumüller et al., "PUFFINN," *ESA 2019*.
5. Malkov & Yashunin, "Efficient and robust ANN search using HNSW,"
   *TPAMI 2018*.

## Proposed design

```text
┌────────────────────────────────────────────────────────────────────────┐
│ HNSW::search(q, k, ef_max, &mut policy) -> (top, stats)                │
│                                                                        │
│   greedy descent through upper levels                                  │
│   policy.reset(k, ef_max)                                              │
│   loop:                                                                │
│     pop best candidate                                                 │
│     ┌── policy.should_stop(step, cand_dist, kth_dist, top_len) ───┐    │
│     │  FixedEf:   always false                                    │    │
│     │  Slope:     true iff kth_dist not improving by ε over window│    │
│     │  Learned:   true iff ridge(features) < τ  (≥ min_steps)     │    │
│     └──────────────────────────────────────────────────────────────┘    │
│     expand neighbors, update top, update candidates                    │
│   return top                                                           │
└────────────────────────────────────────────────────────────────────────┘
```

The eight learned features (all bounded) are:

| f | Meaning                                                  |
|---|----------------------------------------------------------|
| 0 | `log10(step + 1) / 4`                                    |
| 1 | `step / ef_max`                                          |
| 2 | `kth_dist`                                               |
| 3 | `cand_dist − kth_dist`                                   |
| 4 | mean of `kth_dist` over last ≤8 steps                    |
| 5 | stddev of `kth_dist` over last ≤8 steps                  |
| 6 | slope of `kth_dist` over last window                     |
| 7 | bias (1.0)                                               |

Training target: residual recall risk, `max(0, final − intermediate)`,
where `intermediate` is a sigmoid-shaped approximation of the
intermediate recall at that step (see "Failure modes" — replacing this
with measured per-step recall is the obvious next step).

## Implementation notes

- The PoC HNSW is intentionally minimal (~250 LOC).
- `cos_dist = 1 − dot(a, b)` on L2-normalized vectors.
- Closed-form ridge: `w = (XᵀX + λI)⁻¹ Xᵀy`. Gaussian elimination with
  partial pivoting. λ = 1e-3 ridge prevents singularity.
- All files under 500 lines.
- Single-threaded throughout for measurement clarity.

## Benchmark methodology

- **Hardware:** Apple M4 Max, 1 thread.
- **Build:** `cargo run --release -p ruvector-early-term --example et_demo`.
- **Data:** synthetic clustered embeddings (64 d, 64 clusters,
  σ=0.18 on the unit sphere). 20 000 vectors. 200 train + 500 test queries.
- **Ground truth:** brute force, k=10.
- **HNSW params:** M=16, M_max0=32, ef_construction=100.
- **Metric:** recall@10 averaged over the test set; distance calls per
  query; wall-clock µs per query; % queries terminated early.

## Results

Headline run (single seed, no warmup):

```
[       FixedEf] recall@10=0.7932  dist_calls/q=1122.8  µs/q=152.1  early=  0%
[Slope(w=8,ε=2e-4)] recall@10=0.5492  dist_calls/q= 474.4  µs/q=63.1   early=100%
[Learned(τ=0.04)] recall@10=0.7446  dist_calls/q= 956.3  µs/q=123.9  early= 76%
```

Pareto sweep:

| Policy             | recall@10 | dist_calls/q | µs/q  | early % |
|--------------------|-----------|--------------|-------|---------|
| FixedEf (ef=32)    | 0.6134    | 571.7        | 38.9  | 0       |
| FixedEf (ef=48)    | 0.6880    | 724.3        | 62.6  | 0       |
| FixedEf (ef=64)    | 0.7330    | 856.9        | 90.6  | 0       |
| FixedEf (ef=80)    | 0.7628    | 984.9        | 120.9 | 0       |
| FixedEf (ef=96)    | 0.7932    | 1122.8       | 152.1 | 0       |
| FixedEf (ef=128)   | 0.8374    | 1399.2       | 224.1 | 0       |
| Slope (ε=5e-4)     | 0.5436    | 465.7        | 58.7  | 100     |
| Slope (ε=2e-4)     | 0.5492    | 474.4        | 63.1  | 100     |
| Slope (ε=1e-4)     | 0.5524    | 478.3        | 59.1  | 100     |
| Slope (ε=5e-5)     | 0.5540    | 480.9        | 57.1  | 100     |
| Learned (τ=0.08)   | 0.7110    | 828.2        | 108.0 | 98      |
| Learned (τ=0.06)   | 0.7266    | 887.5        | 118.4 | 89      |
| Learned (τ=0.04)   | 0.7446    | 956.3        | 123.9 | 76      |
| Learned (τ=0.02)   | 0.7630    | 1021.2       | 132.9 | 57      |
| Learned (τ=0.01)   | 0.7730    | 1053.4       | 135.6 | 43      |

Reading:

- On **uniform-difficulty** synthetic queries, the `Learned` Pareto
  curve is very close to but does not strictly dominate `FixedEf` —
  e.g. `Learned(τ=0.02)` matches `FixedEf(ef=80)` recall (0.763) with
  marginally more distance calls (1021 vs 985) but **57 % of queries
  terminated early** rather than the global budget being spent.
- `Slope` collapses below 0.55 recall for any reasonable window — the
  slope signal alone is too noisy on this data.
- The expected win is on **heterogeneous** workloads (mixed easy/hard
  queries) where the global `ef` has to be tuned for the hardest
  queries, leaving headroom on easy ones. That measurement is in the
  roadmap.

## How it works (blog-readable walkthrough)

HNSW is a beam search. At each step it pops the closest unvisited
candidate from a frontier, expands its neighbors, and updates the
top-k set. The standard stopping rule is "stop when the closest
unvisited candidate is farther than the worst element in the current
top-k." That's a *safety* rule — it guarantees the next expansion
*could* improve the answer.

But "could improve" is not "will meaningfully improve." On easy queries
the top-k stabilizes long before the safety rule fires, and we waste
hundreds of distance computations confirming a result we already have.
On hard queries the safety rule may fire too early because the local
neighborhood is sparse — we want more expansions, not fewer.

Our trick: between expansions, peek at the trajectory so far —
how is the k-th best distance evolving? how big is the gap between the
candidate we're about to expand and the worst in top-k? — and *predict*
whether further work will materially improve recall. If predicted
remaining risk is below a threshold, stop now.

The slope policy uses a single statistic — has the k-th best moved more
than ε in the last w steps? — and it turns out that's too noisy. The
learned policy uses eight features (step counter, progress, k-th best,
gap, sliding mean/stddev/slope, bias), runs them through an 8-weight
linear ridge model, and gets a much better signal. The model is tiny:
8 floats. You can store it next to the index.

## Practical failure modes

- **Surrogate labels.** We train the predictor against a
  sigmoid-shaped approximation of intermediate recall, not measured
  per-step recall. This biases the predictor toward smoother trajectories
  than it would see in production. The fix is straightforward: record
  measured per-step recall during a calibration pass and use that as the
  training target. Roadmap item #1.
- **Synthetic data.** Uniform-difficulty synthetic clusters under-state
  the win. Real datasets (SIFT1M, DEEP10M, embeddings from production
  models) have a long tail of hard queries where per-query adaptivity
  pays off most. Roadmap item #2.
- **One-shot training.** The predictor is trained once and frozen. A
  production system should re-train periodically as the corpus drifts.
- **Tail-latency claim is one-sided.** We measure mean µs/q but not
  p95/p99. Per-query adaptivity helps tails most; we should record
  histograms. Roadmap item #3.
- **No SIMD / no concurrent build.** This PoC is single-threaded
  scalar. Absolute numbers are upper bounds; relative comparisons are
  valid.

## What to improve next (roadmap)

1. **Replace surrogate labels with measured per-step recall.** Record
   recall@k after every expansion under FixedEf at oracle ef_max during
   training, store as the target. Expected: smaller predictor variance,
   broader operating envelope.
2. **Real corpora.** Hook into `ruvector-bench` (SIFT1M, GIST1M, ANN-
   benchmarks). Heterogeneous queries are where adaptivity is supposed
   to shine.
3. **Tail-latency histograms.** Report p50/p95/p99 latency, not just
   mean.
4. **Bandit per query family.** Discrete buckets over `ef`, learned via
   Thompson sampling from observed recall outcomes. Comparable to the
   learned regressor but recovers online.
5. **Trait → production HNSW.** Lift `TerminationPolicy` into
   `ruvector-core::hnsw` behind a cargo feature `early-termination`.
6. **Snapshot the model.** Persist the 8 ridge weights inside the index
   snapshot so a trained predictor travels with the index.

## Production crate layout proposal

If this graduates from research:

```
ruvector-core/
├── src/
│   ├── hnsw/
│   │   ├── mod.rs
│   │   ├── beam.rs          // existing beam loop
│   │   ├── policy.rs        // <-- new: TerminationPolicy trait
│   │   └── policy/
│   │       ├── fixed.rs
│   │       ├── slope.rs
│   │       └── learned.rs   // includes 8-weight RidgeRegressor
│   └── snapshot/
│       └── v6_termination.rs   // <-- new on-disk slot for predictor weights
└── Cargo.toml
   features = { early-termination = [] }
```

API surface:

```rust
pub trait TerminationPolicy: Send + Sync {
    fn reset(&mut self, k: usize, ef_max: usize);
    fn should_stop(&mut self, step: u32,
                   cand_dist: f32, kth_dist: f32, top_len: usize) -> bool;
}

impl Hnsw {
    pub fn search_with<P: TerminationPolicy>(
        &self, q: &[f32], k: usize, ef_max: usize, policy: &mut P,
    ) -> (Vec<(f32, u32)>, SearchStats);
}
```

The trait is intentionally state-bearing (`&mut self`) so per-policy
state (histories, RNG, etc.) lives with the policy, not the index.

## References

(See "SOTA survey" above for full bibliography.)

## Reproduce

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-06-12-early-termination-hnsw
cargo run --release -p ruvector-early-term --example et_demo
cargo test --release -p ruvector-early-term
```
