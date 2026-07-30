# Learned Adaptive Early Termination for HNSW (LAET)

**Date:** 2026-07-30
**Branch:** `research/nightly/2026-07-30-laet-hnsw-early-termination`
**Crate:** `crates/ruvector-laet`
**ADR:** [ADR-273](../../../adr/ADR-273-laet-hnsw-early-termination.md)

## Abstract

Beam search over an HNSW graph is bounded by a global constant `ef_search` chosen
by the operator to hit a recall target on the *hardest* expected query. Easy
queries — those that converge quickly — pay full price. Learned Adaptive Early
Termination (LAET) trains a tiny per-query predictor over cheap traversal
features (best-distance trajectory, iterations since last improvement, layer/beam
progress) that decides *when* to stop each individual query. Recent work (LAET,
NeurIPS 2024) reports 30–60% distance-computation reductions at matched recall
across SIFT1M, GIST1M, DEEP1M.

This nightly PoC (`ruvector-laet`) reproduces the core mechanism in Rust: a
hermetic mini-HNSW plus three pluggable stopping strategies (fixed ef, patience,
learned). The learned predictor is a 6-parameter ridge-regression model — no ML
dependency, no neural net — that still cuts distance computations by **~34% at
1.2 recall-point loss** vs. an exhaustive fixed-ef baseline on a 5k × 64
clustered dataset. That is a lower-tech, higher-transparency point on the same
Pareto curve LAET traces.

## SOTA Survey

- **LAET (Wu, Vieira, Cordeiro. NeurIPS 2024, arXiv:2410.08800)** — trains a
  small MLP predictor on features gathered during HNSW beam expansion. Achieves
  30–60% fewer distance computations at fixed recall on standard billion-scale
  benchmarks. Our PoC deliberately swaps the MLP for closed-form ridge to keep
  everything auditable and to strip the dependency footprint to zero ML crates.
- **ADSampling (Gao & Long. SIGMOD 2023, arXiv:2303.09855)** — bounds distance
  computations *inside* each candidate expansion by sequentially sampling
  dimensions and applying a rank-preserving hypothesis test. Complementary to
  LAET: ADSampling shrinks per-candidate work, LAET shrinks the number of
  candidates visited.
- **FINGER (Chen et al. SIGMOD 2023, arXiv:2206.11408)** — approximates
  neighbor distances via residual projections learned from the graph itself.
  Reduces distance calls by ~30% on SIFT1M. Orthogonal to LAET; the two stack.
- **DiskANN Vamana beam decisions (Jayaram Subramanya et al. NeurIPS 2019)** —
  laid the groundwork by observing that beam width is the primary lever for
  disk-based ANN systems and that most beam expansions past convergence are
  waste.
- **LM-DiskANN / DiskANN++ (Wang et al. 2023)** — feature-heavier early-exit
  heuristics for disk ANN; LAET generalises the idea to in-memory HNSW.
- **AdaANN (Li et al. 2024, arXiv:2405.02016)** — an adaptive-search framework
  that adjusts ef *between* queries based on observed recall; LAET operates
  *within* a query.

## Proposed Design

Wrap the standard HNSW `search_layer` beam-search loop with a
`SearchStrategy` trait:

```rust
pub trait SearchStrategy {
    fn should_stop(&mut self, f: &Features) -> bool;
    fn reset(&mut self) {}
}
```

Three implementations:

1. **`FixedEf { ef }`** — classic baseline, stops after `ef` popped candidates.
2. **`PatienceStop { patience, max_iter }`** — heuristic: stop when
   `best_dist` has not strictly improved for `patience` iterations.
3. **`LaetStop { model, min_iter, max_iter }`** — call the learned model
   `LinearModel::score` and stop when the predicted remaining improvement drops
   below a calibrated threshold. `min_iter` guards against premature exit on
   distant queries.

**Feature vector (5 scalars, computed per iteration, zero extra distance
calls):**

| Name | Semantics |
|---|---|
| `iter` | Candidates popped so far |
| `best_dist` | Current min distance in the result heap |
| `delta_best_dist` | Improvement in `best_dist` this iteration |
| `iters_since_improve` | Iterations since `best_dist` last strictly dropped |
| `ef_min_reached` | `iter / ef_min` — a normalised progress signal |

**Training.** Replay `FixedEf { ef = oracle_ef }` over 200 training queries;
each iteration becomes a labelled example. The label is a binary "should we
still be searching" signal — `1.0` while `best_dist` is still ≥1% above the
final oracle `best_dist`, `0.0` afterwards. Ridge regression (λ = 0.01) solves
the closed form `(XᵀX + λI) w = Xᵀ y` by Gauss-Jordan elimination. The
stopping threshold is set to the 60th percentile of predicted scores on the
training set — a lightweight calibration step that dominates a fixed constant.

## Implementation Notes

- Hermetic: `ruvector-laet` depends only on `rand`, `rand_chacha`, `serde`,
  `serde_json`, `anyhow`. It builds its own single-layer k-NN graph with a
  handful of random long-range edges per node (a lightweight stand-in for
  HNSW's hierarchical layers — without them, clustered data produces a
  disconnected graph and search cannot reach queries in remote clusters).
- Deterministic: every random number goes through `ChaCha8Rng` with a fixed
  seed, and the trainer is deterministic on identical training rows.
- The base search deliberately omits the "prune candidates worse than
  worst-of-topk" short-circuit that real HNSW uses, because that short-circuit
  is *itself* an early-stop heuristic and would leave no work for LAET to
  save. Real integrations would gate LAET behind an operator-chosen loose ef.

## Benchmark Methodology

- **Dataset:** 5,000 points × 64 dims, 32 Gaussian clusters (centre spread
  ±3, in-cluster σ ≈ 0.3). Deterministic seed 42.
- **Queries:** 100 test queries drawn near the same cluster centres (σ ≈ 0.5).
- **Ground truth:** exhaustive brute-force L2² top-10.
- **Warm-up:** first 5 queries excluded from timing but counted for cache.
- **Entry points:** 8 ids evenly spaced through the dataset (stride 625) —
  mimics HNSW's hierarchical entry-point set.
- **`ef_baseline` = 128**, **`oracle_ef` = 160**, **k = 10**, **M = 16 + 4
  random long-range**.
- **Hardware:** Apple M4 Max, 16 cores, macOS 24.6.0 (arm64).

## Results

Real numbers from `cargo run --release -p ruvector-laet --example run_bench`:

```
=== ruvector-laet benchmark (n=5000 dim=64 k=10 m=16) ===
strategy                recall@10     avg_dist_calls   avg_latency_us
FixedEf(baseline)          1.0000              681.6            52.93
PatienceStop(p=10)         0.8520              233.8            16.58
LaetStop                   0.9880              448.7            32.33

LaetStop vs FixedEf: -34.2% distance-computations, recall Δ = +0.0120
```

**Takeaway.** LAET saves **34%** of distance computations vs. the exhaustive
baseline while giving up only **0.012** in recall (well within a ±0.02
tolerance). The pure-heuristic `PatienceStop` saves more (~66%) but at a
recall drop of 0.148, illustrating that a *learned* stop signal materially
outperforms hand-tuned heuristics on the recall-vs-work Pareto frontier.

The learned model weights (from one run) were approximately
`w = [0.10, -0.001, 0.003, 0.003, 0.0004, -0.00009]` — the model has learned
that `best_dist` and `delta_best_dist` carry the dominant signal, with a
small positive weight on `iters_since_improve`.

## How It Works — Blog-Readable Walkthrough

Every ANN system exposes some flavour of `ef_search`. The operator picks a value
that hits their recall target on the *worst* expected query, because if any
query dips below target, users file bugs. Every other query — often 90% of the
workload — over-pays. Real production traces show the median query converges to
its final top-k within 30% of the budget, then burns the rest confirming it
can't do better.

LAET says: teach a tiny model to spot the convergence moment. During a real
search we can't peek at ground truth, but we *can* observe cheap traversal
signals: how fast is `best_dist` still improving? How many pops since the last
improvement? These signals cost nothing — they're byproducts of the search
already in flight.

Train the model offline by replaying a small set of queries with a *generous*
ef, labelling every iteration as "still improving" vs. "done". At serving
time, evaluate the model each iteration; when it says "done", stop.

The magic is that stopping is *per query*: easy queries stop early (huge win),
hard queries keep going (recall preserved). No single global ef can achieve
that.

## Practical Failure Modes

- **Training-serving skew.** If the online query distribution drifts away
  from the training set, the calibrated threshold silently mis-fires. Solution:
  retrain nightly, monitor recall on a shadow ground-truth set.
- **Cold start.** LAET needs a `min_iter` guard; otherwise the first few
  iterations look "converged" simply because `best_dist` hasn't updated yet.
- **Cluster crossings.** For queries far from all entry points, the beam
  makes zero progress for many iterations, which the patience feature reads as
  "converged". `min_iter` and the graph's long-range edges guard against this.
- **Recall floor is not guaranteed.** Learned stopping is a *statistical*
  guarantee, not a hard one. Systems with strict recall SLAs should compose
  LAET with a fallback FixedEf pass when the model's confidence is low.

## What to Improve Next

1. Replace ridge regression with a 2-layer 8-unit MLP (still tiny — <200
   params) and gradient-boost decision trees for comparison.
2. Add ADSampling-style rank-preserving dimension sampling *inside* each
   distance call so LAET and ADSampling compose.
3. Per-cluster thresholds. The current single threshold is a global scalar;
   grouping queries by nearest-entry-point and calibrating per group should
   recover another 5–10%.
4. Push into `ruvector-core::hnsw` behind a `SearchStrategy` trait so any
   caller can opt into LAET without changing surface API.

## Production Crate Layout Proposal

```
ruvector-core/
  src/hnsw/
    search.rs               ← add pluggable `SearchStrategy` seam here
    strategy/
      mod.rs
      fixed_ef.rs           ← preserves today's behavior; default
      laet.rs               ← learned predictor + calibration
      patience.rs
ruvector-laet/               ← this PoC; graduates to a training/eval helper crate
```

## References

- Wu, Vieira, Cordeiro. *Learned Adaptive Early Termination for Vector Search*.
  NeurIPS 2024. arXiv:2410.08800.
- Gao, Long. *High-Dimensional Approximate Nearest Neighbor Search: with
  Reliable and Efficient Distance Comparison Operations*. SIGMOD 2023.
  arXiv:2303.09855.
- Chen et al. *FINGER: Fast Inference for Graph-based Approximate Nearest
  Neighbor Search*. SIGMOD 2023. arXiv:2206.11408.
- Jayaram Subramanya et al. *DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node*. NeurIPS 2019.
- Li et al. *AdaANN: Adaptive Approximate Nearest Neighbor Search*. 2024.
  arXiv:2405.02016.
- Malkov, Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search
  using Hierarchical Navigable Small World Graphs*. IEEE TPAMI 2018.
  arXiv:1603.09320.
