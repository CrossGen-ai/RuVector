//! # ruvector-learned-termination
//!
//! Learned early termination for HNSW-style beam search.
//!
//! Standard graph-ANN systems use a fixed `ef_search` budget that must be tuned
//! offline. That fixed budget over-searches easy queries (already converged after
//! a few steps) and under-searches hard queries. Prior adaptive variants use
//! entropy or scalar distance thresholds — see the sister crate
//! `ruvector-entropy-ann` which measured entropy to be a saturating signal on
//! real graph traversals. This crate takes a different tack:
//!
//! A **tiny logistic-regression classifier** is trained (offline, from a small
//! held-out query set) on five cheap runtime features:
//!
//! 1. `best_dist` — current closest distance in the results heap
//! 2. `improve_rate` — recent moving-average decrease of `best_dist` per step
//! 3. `gap_kth` — normalized gap between k-th and (k-1)-th result
//! 4. `steps_norm` — `steps_so_far / ef` (fraction of budget consumed)
//! 5. `frontier_ratio` — `unvisited_neighbours_of_current / degree`
//!
//! The classifier outputs the probability that continuing expansion will
//! change the top-k. When `P(improve) < tau`, we terminate.
//!
//! ## Variants
//!
//! | Variant | Description |
//! |---------|-------------|
//! | [`FixedEfSearch`] | Baseline: fixed `ef`, terminate only on standard HNSW pruning |
//! | [`LearnedTermination`] | Logistic classifier gates termination once per step |
//! | [`OracleTermination`] | Upper-bound reference — stops the moment top-k stops changing |
//!
//! ## Design notes
//!
//! - The classifier is a **five-feature logistic regression** — 6 f32 weights, 24
//!   bytes total. Inference is a single dot-product + sigmoid; ~20ns per step.
//! - Training uses **SGD with logistic loss** over triples
//!   (features_at_step_t, top_k_at_step_t_equals_final_top_k). No external ML
//!   deps — pure Rust, ~120 LOC.
//! - The design is **backend-agnostic**: features are extracted from any beam
//!   search's frontier heap. Wiring into full HNSW is a `Feature::extract` swap.
//!
//! ## Measured PoC outcome
//!
//! Real numbers from `cargo run --release -p ruvector-learned-termination --bin
//! benchmark` on a clustered 2 000-vector, 32-dim corpus (10 clusters,
//! noise=0.2, k_graph=20, ef=80, k=10, 200 test queries drawn from a fresh
//! seed so they land off-cluster and stress the search):
//!
//! | Variant | recall@10 | beam dist calls | median steps | beam speedup |
//! |---------|-----------|-----------------|--------------|--------------|
//! | Fixed(ef=80)        | 0.6630 | 192.0 | 81 | 1.00× |
//! | Learned(tau=0.15)   | 0.6600 | 160.0 | 30 | 1.20× |
//! | Learned(tau=0.30)   | 0.6520 | 146.7 | 23 | 1.31× |
//! | Oracle(patience=3)  | 0.6630 | 176.9 | 81 | 1.09× |
//!
//! Learned(tau=0.15) recovers ~17 % of the Fixed baseline's beam distance
//! computations while keeping recall within 0.003 of Fixed. Learned(tau=0.30)
//! trades a further 8-point recall drop for another 11 % of beam cost. The
//! Oracle bound is modest (1.09×) at this ef because the standard HNSW
//! candidate-heap prune already handles trivial cases; the Learned model
//! delivers savings that meaningfully exceed the Oracle bound because it
//! terminates before the top-k has fully stabilised — a tunable recall/cost
//! knob rather than an oracle-limited one.
//!
//! Median steps drop 60–70 % because the learned gate fires on steps where
//! the model predicts P(top-k will change) < tau; total beam dist calls drop
//! by less because most work happens in the first few expansions of the entry
//! point's dense neighbourhood.

pub mod dataset;
pub mod graph;
pub mod predictor;
pub mod search;

pub use graph::{FlatGraph, GraphConfig};
pub use predictor::{FeatureSnapshot, LogisticPredictor, TrainConfig};
pub use search::{FixedEfSearch, Hit, LearnedTermination, OracleTermination, Searcher, SearchStats};

/// Recall@k: fraction of true top-k found in approximate results.
///
/// Denominator is `min(k, ground_truth.len())`; a searcher that returns fewer
/// than `k` results is penalised, not rewarded.
pub fn recall_at_k(ground_truth: &[usize], results: &[Hit], k: usize) -> f32 {
    let k = k.min(ground_truth.len());
    if k == 0 {
        return 0.0;
    }
    let gt: std::collections::HashSet<usize> = ground_truth[..k].iter().cloned().collect();
    let found = results
        .iter()
        .take(k)
        .filter(|h| gt.contains(&h.id))
        .count();
    found as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use search::Hit;

    #[test]
    fn recall_at_k_perfect() {
        let gt = vec![0usize, 1, 2, 3];
        let results: Vec<Hit> = gt
            .iter()
            .enumerate()
            .map(|(i, &id)| Hit { id, dist: i as f32 })
            .collect();
        assert!((recall_at_k(&gt, &results, 4) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_at_k_partial() {
        let gt = vec![0usize, 1, 2, 3];
        let results = vec![
            Hit { id: 0, dist: 0.1 },
            Hit { id: 1, dist: 0.2 },
            Hit { id: 99, dist: 0.3 },
            Hit { id: 100, dist: 0.4 },
        ];
        let r = recall_at_k(&gt, &results, 4);
        assert!((r - 0.5).abs() < 1e-6, "expected 0.5, got {r}");
    }

    #[test]
    fn recall_at_k_empty_gt_is_zero() {
        let gt: Vec<usize> = vec![];
        let results = vec![Hit { id: 0, dist: 0.0 }];
        assert!(recall_at_k(&gt, &results, 5).abs() < 1e-6);
    }
}
