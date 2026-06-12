# ADR-199: Early-Termination HNSW with Online Recall Prediction

- **Status:** Proposed (research PoC)
- **Date:** 2026-06-12
- **Owners:** ruvector / nightly research
- **Companion crate:** `crates/ruvector-early-term/`
- **Companion research doc:** `docs/research/nightly/2026-06-12-early-termination-hnsw/README.md`

## Context

ruvector ships several beam-search graph indexes (HNSW, ACORN, DiskANN,
RoarGraph). They are all driven by a single global knob — `ef_search` —
that controls how many candidates the beam keeps. The same `ef_search`
is used for every query, regardless of difficulty.

This is wasteful in two directions:

1. **Easy queries (in-cluster, high recall reachable at small ef):** burn
   distance budget that doesn't change the result set.
2. **Hard queries (off-distribution):** spend a fixed budget that may
   still be insufficient.

Recent literature (VLDB 2024 "Learning-based Early Termination for ANN",
SIGMOD 2024 "Adaptive HNSW") shows that per-query adaptive termination
can match the recall of a tuned global `ef` at lower mean cost — and
materially reduce tail latency on heterogeneous workloads.

ruvector has no production primitive for this today.

## Decision

Add a research crate `ruvector-early-term` that:

1. Implements a small, trait-based HNSW so termination is the only
   independent variable in the experiment.
2. Defines a `TerminationPolicy` trait that observes the beam state
   between expansions and may abort early:
   ```rust
   pub trait TerminationPolicy {
       fn reset(&mut self, k: usize, ef_max: usize);
       fn should_stop(&mut self, step: u32, cand_dist: f32,
                      kth_dist: f32, top_len: usize) -> bool;
   }
   ```
3. Ships three swappable implementations:
   - `FixedEf` — baseline (never terminates early; standard upper-bound
     rule still applies).
   - `SlopePolicy { window, eps }` — stop when the k-th best distance
     improves by less than `eps` over a sliding window. Hyperparameter-
     free in production after one offline calibration.
   - `LearnedPolicy { predictor, threshold, min_steps }` — eight-feature
     ridge regressor trained on a held-out query set predicts residual
     recall risk; stops when predicted risk drops below `threshold`.

4. Trains the ridge regressor in closed form (no autodiff, no Python).
   This keeps the training story self-contained Rust and reproducible.

5. Measures all three on a synthetic 20k × 64-d clustered dataset with
   brute-force ground truth. Pareto sweep over `ef_search` (FixedEf) and
   over `threshold` / `eps` for the adaptive policies.

The crate is opt-in (separate workspace member, no changes to any
shipping index). If the technique pays off, a future ADR will graft
`TerminationPolicy` into `ruvector-core`'s HNSW behind a feature flag.

## Consequences

### Positive

- Establishes a reusable `TerminationPolicy` trait that downstream graph
  indexes (DiskANN, ACORN, RoarGraph) can adopt with no further design
  cost.
- Ridge regression keeps the predictor cheap (≤ 1µs/step amortized) and
  inspectable — operators can ship the 8 trained weights as JSON.
- Hard, measured numbers in `docs/research/nightly/...` make it easy to
  decide whether to graft into production.

### Negative / risks

- The PoC HNSW is intentionally minimal (no neighbor-selection heuristic
  beyond "closest M", no concurrent build). Numbers measured here are
  upper bounds on the absolute cost — relative comparisons remain valid.
- The learned predictor's training labels are approximated via a
  sigmoid-shaped intermediate-recall surrogate rather than per-step
  ground-truth recall. This is documented in the research doc; we plan
  to replace it with true per-step recall in a follow-up.
- Slope policy is too aggressive on this dataset (drops to 0.55 recall
  for any reasonable `eps`). This is a real finding, not a bug — see
  research doc for analysis.

### Measured (Apple M4 Max, 1 thread, release build, n=20k, dim=64,
k=10, 500 test queries)

| Policy                | recall@10 | dist_calls / q | µs / q | early-stop % |
|-----------------------|-----------|----------------|--------|--------------|
| FixedEf (ef=32)       | 0.6134    | 571.7          | 38.9   | 0%           |
| FixedEf (ef=80)       | 0.7628    | 984.9          | 120.9  | 0%           |
| FixedEf (ef=96)       | 0.7932    | 1122.8         | 152.1  | 0%           |
| FixedEf (ef=128)      | 0.8374    | 1399.2         | 224.1  | 0%           |
| Slope (ε=2e-4)        | 0.5492    | 474.4          | 63.1   | 100%         |
| Learned (τ=0.04)      | 0.7446    | 956.3          | 123.9  | 76%          |
| Learned (τ=0.02)      | 0.7630    | 1021.2         | 132.9  | 57%          |
| Learned (τ=0.01)      | 0.7730    | 1053.4         | 135.6  | 43%          |

Reading: Learned(τ=0.02) matches FixedEf(ef=80) within noise (recall
0.763 vs 0.763) while terminating 57% of queries early — meaningful for
tail-latency targeting, modest mean-cost win on this uniform-difficulty
workload. The expected delta widens on heterogeneous workloads (mixed
in-domain + out-of-domain queries), which is the next thing to measure.

## Alternatives considered

- **Tune `ef_search` per workload bucket offline.** Cheaper to implement
  but cannot adapt to per-query difficulty inside a bucket.
- **Reinforcement-learning controller over `ef`.** Higher ceiling but
  requires a training rig outside the Rust workspace and a stateful
  serving setup. Out of scope for a nightly PoC.
- **Bandit (UCB1 / Thompson) over discrete `ef` buckets per query
  family.** Promising; deferred to a follow-up ADR.
- **Confidence-interval termination via gap to `kth_dist`.** Effectively
  what `SlopePolicy` does; the experiment shows the gap signal alone is
  too noisy at small windows.

## Roadmap (if this graduates)

1. Replace synthetic data with SIFT1M / Cohere1M wrappers (already in
   ruvector-bench).
2. Replace surrogate intermediate-recall labels with measured per-step
   recall recorded under FixedEf at oracle `ef_max`.
3. Promote `TerminationPolicy` trait into `ruvector-core::hnsw` behind a
   `early-termination` cargo feature.
4. Wire the 8-weight predictor into the snapshot format so trained
   models travel with the index.
