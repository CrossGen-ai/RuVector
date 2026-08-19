//! Termination policies for the beam search.
//!
//! Each policy sees the current top-`k` result set (sorted best-first by
//! distance) after every graph-node expansion and decides whether the search
//! should stop.

use crate::Scored;

/// The generic hook the search loop calls after each expansion.
///
/// `top_k` is passed already sorted by distance ascending, and its length
/// is guaranteed to be `min(k, results_so_far)`.
pub trait TerminationPolicy {
    /// A short, machine-friendly tag for the policy — used by the benchmark
    /// to label rows.
    fn name(&self) -> &'static str;

    /// Called by [`crate::search::search`] after every graph-node expansion
    /// (i.e. once per `visit`). Return `true` to stop the search early.
    ///
    /// `iter` is the 1-based expansion count. `k` is the requested top-k
    /// (identical to `top_k.len()` once the search has warmed up).
    fn should_stop(&mut self, iter: usize, k: usize, top_k: &[Scored]) -> bool;

    /// Reset per-query state. The search loop calls this once before each
    /// query so the same policy instance is reusable across queries.
    fn reset(&mut self);
}

// ────────────────────────────────────────────────────────────────────────────
// Baseline: fixed visit budget.
// ────────────────────────────────────────────────────────────────────────────

/// Never terminates before `max_visits` expansions — this is what a
/// classic `ef_search`-style bound does.
#[derive(Clone, Debug)]
pub struct FixedBudget;

impl TerminationPolicy for FixedBudget {
    fn name(&self) -> &'static str { "fixed" }
    fn should_stop(&mut self, _iter: usize, _k: usize, _top_k: &[Scored]) -> bool { false }
    fn reset(&mut self) {}
}

// ────────────────────────────────────────────────────────────────────────────
// Common heuristic: relative-gap plateau on the k-th distance.
// ────────────────────────────────────────────────────────────────────────────

/// Terminate when the k-th distance has improved by strictly less than
/// `epsilon * dist_k` for `window` consecutive iterations. This is the
/// standard "distance plateau" heuristic reported in several ANN
/// libraries and papers.
#[derive(Clone, Debug)]
pub struct GapThreshold {
    pub epsilon: f32,
    pub window: usize,
    pub min_visits: usize,
    prev_kth: Option<f32>,
    stable_count: usize,
}

impl GapThreshold {
    pub fn new(epsilon: f32, window: usize, min_visits: usize) -> Self {
        assert!(epsilon >= 0.0);
        assert!(window >= 1);
        Self { epsilon, window, min_visits, prev_kth: None, stable_count: 0 }
    }
}

impl TerminationPolicy for GapThreshold {
    fn name(&self) -> &'static str { "gap" }
    fn should_stop(&mut self, iter: usize, k: usize, top_k: &[Scored]) -> bool {
        if iter < self.min_visits || top_k.len() < k {
            return false;
        }
        let cur = top_k[k - 1].dist;
        match self.prev_kth {
            None => {
                self.prev_kth = Some(cur);
                self.stable_count = 0;
                false
            }
            Some(prev) => {
                let improved = prev - cur; // >= 0 since results only get better
                let plateau = improved < self.epsilon * prev.max(1e-12);
                self.stable_count = if plateau { self.stable_count + 1 } else { 0 };
                self.prev_kth = Some(cur);
                self.stable_count >= self.window
            }
        }
    }
    fn reset(&mut self) {
        self.prev_kth = None;
        self.stable_count = 0;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// This crate's contribution: top-k ordering stability under Kendall's tau.
// ────────────────────────────────────────────────────────────────────────────

/// Terminate when Kendall's tau between the current top-k id list and the
/// top-k list from `window` iterations ago is at least `tau_threshold`, for
/// `stable_iters` consecutive iterations.
///
/// The intuition:
///
///   * The gap-threshold heuristic looks at a *scalar* — the k-th distance —
///     which is noisy: a single new candidate can drop the k-th distance
///     without changing which items are actually returned. That triggers
///     false "still improving" signals and wastes visits.
///   * Kendall's tau on the *id ranking* looks at what the caller actually
///     receives. If the top-k identities and their order have not moved for
///     a few iterations, additional expansions statistically don't matter.
///
/// This is a rank-agreement stopping rule, not a recall estimator. It gives
/// up "unbiased predicted recall" (which policies like adaptive-recall-ann
/// try to produce) in exchange for zero calibration and no LUT.
#[derive(Clone, Debug)]
pub struct KendallTauStability {
    pub tau_threshold: f32,
    pub window: usize,
    pub stable_iters: usize,
    pub min_visits: usize,
    prev_ids: Vec<u32>,
    stable_count: usize,
    since_last_check: usize,
}

impl KendallTauStability {
    pub fn new(tau_threshold: f32, window: usize, stable_iters: usize, min_visits: usize) -> Self {
        assert!((0.0..=1.0).contains(&tau_threshold));
        assert!(window >= 1);
        assert!(stable_iters >= 1);
        Self {
            tau_threshold,
            window,
            stable_iters,
            min_visits,
            prev_ids: Vec::new(),
            stable_count: 0,
            since_last_check: 0,
        }
    }

    /// Kendall's tau on two same-length permutations of ids. Concordant pairs
    /// minus discordant pairs, divided by n(n-1)/2. If the two id sets are
    /// not identical we penalize: pairs involving ids present in only one
    /// list are counted as discordant. Runs in O(k^2), fine for typical
    /// k (≤ 100).
    pub fn tau(a: &[u32], b: &[u32]) -> f32 {
        let n = a.len();
        if n < 2 || b.len() != n {
            // With <2 elements tau is undefined; return 1.0 (trivially stable).
            return if a == b { 1.0 } else { 0.0 };
        }
        // k is small (typically ≤ 100), a linear index-of is fine and
        // keeps this dependency-free.
        let index_in = |slice: &[u32], id: u32| slice.iter().position(|&x| x == id);

        let mut concordant: i64 = 0;
        let mut discordant: i64 = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let ai = a[i];
                let aj = a[j];
                let bi = index_in(b, ai);
                let bj = index_in(b, aj);
                match (bi, bj) {
                    (Some(bi), Some(bj)) => {
                        // a-order: i<j means a[i] ranked higher than a[j].
                        // b-order concordant iff bi<bj.
                        if bi < bj { concordant += 1; } else { discordant += 1; }
                    }
                    _ => {
                        // Missing id — count as discordant (penalize churn).
                        discordant += 1;
                    }
                }
            }
        }
        let total = (n * (n - 1) / 2) as f32;
        (concordant - discordant) as f32 / total
    }
}

impl TerminationPolicy for KendallTauStability {
    fn name(&self) -> &'static str { "kendall" }
    fn should_stop(&mut self, iter: usize, k: usize, top_k: &[Scored]) -> bool {
        if iter < self.min_visits || top_k.len() < k {
            return false;
        }
        self.since_last_check += 1;
        if self.since_last_check < self.window {
            return false;
        }
        self.since_last_check = 0;

        let cur_ids: Vec<u32> = top_k.iter().map(|s| s.id).collect();
        if self.prev_ids.is_empty() {
            self.prev_ids = cur_ids;
            self.stable_count = 0;
            return false;
        }
        let tau = Self::tau(&self.prev_ids, &cur_ids);
        if tau >= self.tau_threshold {
            self.stable_count += 1;
        } else {
            self.stable_count = 0;
        }
        self.prev_ids = cur_ids;
        self.stable_count >= self.stable_iters
    }
    fn reset(&mut self) {
        self.prev_ids.clear();
        self.stable_count = 0;
        self.since_last_check = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(id: u32, d: f32) -> Scored { Scored { id, dist: d } }

    #[test]
    fn fixed_never_stops() {
        let mut p = FixedBudget;
        for it in 1..1000 {
            assert!(!p.should_stop(it, 5, &[s(1,0.0);5]));
        }
    }

    #[test]
    fn gap_triggers_on_plateau() {
        let mut p = GapThreshold::new(0.01, 3, 0);
        let mut kth = 1.0f32;
        let mut stopped_at = None;
        for it in 1..50 {
            let top: Vec<Scored> = (0..5).map(|i| s(i, kth)).collect();
            if p.should_stop(it, 5, &top) { stopped_at = Some(it); break; }
            // Simulate rapid decrease for first 5 iters, then plateau.
            if it < 5 { kth *= 0.5; }
        }
        assert!(stopped_at.is_some(), "gap policy should terminate on plateau");
    }

    #[test]
    fn gap_does_not_trigger_while_improving() {
        let mut p = GapThreshold::new(0.01, 3, 0);
        let mut kth = 1.0f32;
        for it in 1..20 {
            let top: Vec<Scored> = (0..5).map(|i| s(i, kth)).collect();
            let stop = p.should_stop(it, 5, &top);
            assert!(!stop, "should not stop while improving");
            kth *= 0.5; // always >epsilon improvement
        }
    }

    #[test]
    fn kendall_tau_identity_is_one() {
        // Distinct ids — a top-k id list is always a set, never a bag.
        let a = vec![3u32, 1, 4, 7, 5, 9, 2, 6];
        let b = a.clone();
        assert!((KendallTauStability::tau(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn kendall_tau_reverse_is_minus_one() {
        let a = vec![1u32, 2, 3, 4, 5];
        let b: Vec<u32> = a.iter().rev().cloned().collect();
        let t = KendallTauStability::tau(&a, &b);
        assert!((t + 1.0).abs() < 1e-5, "reverse tau = {t}");
    }

    #[test]
    fn kendall_policy_triggers_when_stable() {
        let mut p = KendallTauStability::new(0.95, 1, 3, 0);
        let stable = vec![
            Scored{id:1,dist:0.1}, Scored{id:2,dist:0.2}, Scored{id:3,dist:0.3},
            Scored{id:4,dist:0.4}, Scored{id:5,dist:0.5},
        ];
        let mut fired = false;
        for it in 1..50 {
            if p.should_stop(it, 5, &stable) { fired = true; break; }
        }
        assert!(fired);
    }

    #[test]
    fn kendall_policy_reset_clears_state() {
        let mut p = KendallTauStability::new(0.95, 1, 2, 0);
        let stable = vec![Scored{id:1,dist:0.1}, Scored{id:2,dist:0.2}, Scored{id:3,dist:0.3}];
        for it in 1..5 { p.should_stop(it, 3, &stable); }
        p.reset();
        assert_eq!(p.stable_count, 0);
        assert!(p.prev_ids.is_empty());
    }
}
