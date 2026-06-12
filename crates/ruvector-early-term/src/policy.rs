//! Termination policies. Each is invoked once per beam-search step with:
//!   - `step`: 1-based step counter
//!   - `cand_dist`: distance of the candidate about to be expanded
//!   - `kth_dist`: current distance of the k-th best (∞ if fewer than k)
//!   - `top_len`: number of items currently in top set
//!
//! `should_stop` returns true to abort the search.

use crate::predictor::{QueryFeatures, RidgeRegressor};

pub trait TerminationPolicy {
    fn reset(&mut self, k: usize, ef_max: usize);
    fn should_stop(&mut self, step: u32, cand_dist: f32, kth_dist: f32, top_len: usize) -> bool;
}

/// Baseline: never terminate early. Search runs until standard upper-bound
/// rule fires inside the HNSW loop (`cand > upper && top >= ef`).
pub struct FixedEf;
impl TerminationPolicy for FixedEf {
    fn reset(&mut self, _k: usize, _ef_max: usize) {}
    fn should_stop(&mut self, _s: u32, _c: f32, _k: f32, _t: usize) -> bool { false }
}

/// Stop if the k-th best distance has not improved by more than `eps`
/// over a sliding window of `window` steps, AND we already have k items.
pub struct SlopePolicy {
    pub window: u32,
    pub eps: f32,
    k: usize,
    history: Vec<f32>,
}
impl SlopePolicy {
    pub fn new(window: u32, eps: f32) -> Self {
        Self { window, eps, k: 0, history: Vec::with_capacity(64) }
    }
}
impl TerminationPolicy for SlopePolicy {
    fn reset(&mut self, k: usize, _ef_max: usize) {
        self.k = k;
        self.history.clear();
    }
    fn should_stop(&mut self, _step: u32, _cand: f32, kth: f32, top_len: usize) -> bool {
        if top_len < self.k || !kth.is_finite() { return false; }
        self.history.push(kth);
        let n = self.history.len();
        if (n as u32) < self.window + 1 { return false; }
        let past = self.history[n - 1 - self.window as usize];
        let improvement = past - kth;
        improvement < self.eps
    }
}

/// Online learned predictor. At each step it builds a `QueryFeatures` vector
/// describing the search trajectory so far and asks the ridge regressor for
/// an expected residual-recall risk. If the predicted risk is below
/// `risk_threshold` (e.g. 0.05 = predicted recall already ≥ 0.95), stop.
pub struct LearnedPolicy {
    pub predictor: RidgeRegressor,
    pub risk_threshold: f32,
    pub min_steps: u32,
    k: usize,
    ef_max: usize,
    history: Vec<f32>,
    cand_history: Vec<f32>,
}

impl LearnedPolicy {
    pub fn new(predictor: RidgeRegressor, risk_threshold: f32, min_steps: u32) -> Self {
        Self {
            predictor,
            risk_threshold,
            min_steps,
            k: 0,
            ef_max: 0,
            history: Vec::with_capacity(128),
            cand_history: Vec::with_capacity(128),
        }
    }
}

impl TerminationPolicy for LearnedPolicy {
    fn reset(&mut self, k: usize, ef_max: usize) {
        self.k = k;
        self.ef_max = ef_max;
        self.history.clear();
        self.cand_history.clear();
    }
    fn should_stop(&mut self, step: u32, cand: f32, kth: f32, top_len: usize) -> bool {
        if top_len < self.k || !kth.is_finite() { return false; }
        self.history.push(kth);
        self.cand_history.push(cand);
        if step < self.min_steps { return false; }
        let feats = QueryFeatures::from_history(step, cand, kth, &self.history, &self.cand_history, self.ef_max);
        let risk = self.predictor.predict(&feats.to_vec());
        risk < self.risk_threshold
    }
}
