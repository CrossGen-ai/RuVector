//! Three `ef_search` controllers.
//!
//! Feature vector used by all learned/heuristic variants:
//!   x = [1, d_entry, d1, d2, d2/d1, first_hop_mean]
//!
//! All features come from a *cheap probe* (small fixed ef) — the same call is
//! made once per query and consumed by the controller to pick the real ef.

use crate::hnsw::ProbeStats;

pub trait EfController: Send + Sync {
    fn name(&self) -> &'static str;
    /// Pick an ef given probe features. `k` is the requested top-k so the
    /// controller can honor `ef >= k`.
    fn choose_ef(&self, probe: &ProbeStats, k: usize) -> usize;
}

/// Constant `ef` — the baseline everyone compares against.
pub struct FixedEf { pub ef: usize }
impl EfController for FixedEf {
    fn name(&self) -> &'static str { "fixed" }
    fn choose_ef(&self, _probe: &ProbeStats, k: usize) -> usize { self.ef.max(k) }
}

/// Gap-ratio heuristic. Idea: when d2/d1 is close to 1 (dense cluster,
/// ambiguous top result) we need a large ef; when d2/d1 is large (isolated
/// top result), a small ef is enough.
///
/// `ef = clip(base * (1 + alpha / max(gap - 1, eps)), min_ef, max_ef)`
pub struct GapRatioEf {
    pub base: f32,
    pub alpha: f32,
    pub min_ef: usize,
    pub max_ef: usize,
}
impl Default for GapRatioEf {
    fn default() -> Self {
        GapRatioEf { base: 48.0, alpha: 1.2, min_ef: 32, max_ef: 384 }
    }
}
impl EfController for GapRatioEf {
    fn name(&self) -> &'static str { "gap_ratio" }
    fn choose_ef(&self, probe: &ProbeStats, k: usize) -> usize {
        let d1 = probe.d1.max(1e-6);
        let gap = probe.d2 / d1;
        let extra = self.alpha / (gap - 1.0).max(0.05);
        let raw = self.base * (1.0 + extra);
        (raw as usize).clamp(self.min_ef, self.max_ef).max(k)
    }
}

/// Learned linear predictor. Fit an ordinary least squares model to predict
/// the *minimum* ef needed to hit target recall on a small calibration set.
/// Prediction: `y = w · x`, then clamped and rounded.
///
/// Closed-form OLS: `w = (X^T X)^-1 X^T y`. Implemented with Gauss-Jordan
/// so we stay dep-free.
pub struct LearnedLinearEf {
    weights: [f32; 6],
    min_ef: usize,
    max_ef: usize,
}

impl LearnedLinearEf {
    pub fn new(weights: [f32; 6], min_ef: usize, max_ef: usize) -> Self {
        LearnedLinearEf { weights, min_ef, max_ef }
    }

    /// Compute the 6-dim feature vector for a probe.
    pub fn features(probe: &ProbeStats) -> [f32; 6] {
        let d1 = probe.d1.max(1e-6);
        let gap = probe.d2 / d1;
        [1.0, probe.d_entry, probe.d1, probe.d2, gap, probe.first_hop_mean]
    }

    /// Fit OLS given (X, y). Returns `Self` on success.
    pub fn fit(xs: &[[f32; 6]], ys: &[f32], min_ef: usize, max_ef: usize) -> Option<Self> {
        let n = xs.len();
        if n < 6 || n != ys.len() { return None; }
        // Build 6x6 XtX and 6x1 Xty.
        let mut a = [[0.0f64; 7]; 6]; // augmented [XtX | Xty]
        for row in 0..n {
            let x = xs[row];
            let y = ys[row] as f64;
            for i in 0..6 {
                for j in 0..6 {
                    a[i][j] += (x[i] as f64) * (x[j] as f64);
                }
                a[i][6] += (x[i] as f64) * y;
            }
        }
        // Ridge for stability.
        for i in 0..6 { a[i][i] += 1e-3; }
        // Gauss-Jordan.
        for pivot in 0..6 {
            let mut max_row = pivot;
            for r in (pivot + 1)..6 {
                if a[r][pivot].abs() > a[max_row][pivot].abs() { max_row = r; }
            }
            if a[max_row][pivot].abs() < 1e-12 { return None; }
            a.swap(pivot, max_row);
            let pv = a[pivot][pivot];
            for c in 0..7 { a[pivot][c] /= pv; }
            for r in 0..6 {
                if r == pivot { continue; }
                let factor = a[r][pivot];
                if factor == 0.0 { continue; }
                for c in 0..7 { a[r][c] -= factor * a[pivot][c]; }
            }
        }
        let mut w = [0f32; 6];
        for i in 0..6 { w[i] = a[i][6] as f32; }
        Some(LearnedLinearEf { weights: w, min_ef, max_ef })
    }

    pub fn weights(&self) -> [f32; 6] { self.weights }
}

impl EfController for LearnedLinearEf {
    fn name(&self) -> &'static str { "learned_linear" }
    fn choose_ef(&self, probe: &ProbeStats, k: usize) -> usize {
        let feats = Self::features(probe);
        let mut y = 0.0f32;
        for i in 0..6 { y += self.weights[i] * feats[i]; }
        // Round up (we want to hit recall from above).
        let ef = y.ceil() as i64;
        let ef = if ef < 0 { self.min_ef as i64 } else { ef };
        (ef as usize).clamp(self.min_ef, self.max_ef).max(k)
    }
}
