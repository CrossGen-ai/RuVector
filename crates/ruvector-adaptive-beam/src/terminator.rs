//! Pluggable beam-termination strategies for adaptive-beam HNSW.

use crate::p2::P2Quantile;

/// Per-query statistics returned by a search.
#[derive(Debug, Clone, Default)]
pub struct TerminationStats {
    /// Number of distance evaluations performed at layer 0.
    pub distance_evals: u64,
    /// Number of beam expansions at layer 0.
    pub expansions: u64,
    /// Number of times an expansion improved the top-k.
    pub improvements: u64,
    /// Whether termination fired before reaching ef_max.
    pub early_stopped: bool,
}

/// Strategy that decides when to stop expanding the HNSW beam.
///
/// `min_unexpanded` is the lowest distance among candidates in the
/// candidate heap not yet expanded; `worst_topk` is the largest distance
/// currently in the result heap; `improved_this_step` indicates whether
/// the last expansion produced a new top-k member.
pub trait BeamTerminator: Send + Sync {
    fn reset(&mut self);
    fn should_stop(
        &mut self,
        expansions: u64,
        min_unexpanded: f32,
        worst_topk: f32,
        improved_this_step: bool,
    ) -> bool;
    /// Size of the result pool maintained during layer-0 search.
    /// HNSW classically uses `ef_search`. For adaptive terminators this
    /// is the upper bound (worst-case ef).
    fn ef_pool(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Classical HNSW: stop when `min_unexpanded > worst_topk` (handled by the
/// search loop) or when `expansions >= ef_search`.
#[derive(Debug, Clone)]
pub struct FixedEfTerminator {
    pub ef_search: u64,
}
impl FixedEfTerminator {
    pub fn new(ef_search: usize) -> Self {
        Self {
            ef_search: ef_search as u64,
        }
    }
}
impl BeamTerminator for FixedEfTerminator {
    fn reset(&mut self) {}
    fn should_stop(
        &mut self,
        expansions: u64,
        _min_unexpanded: f32,
        _worst_topk: f32,
        _improved: bool,
    ) -> bool {
        expansions >= self.ef_search
    }
    fn ef_pool(&self) -> usize {
        self.ef_search as usize
    }
    fn name(&self) -> &'static str {
        "fixed_ef"
    }
}

/// Stop when `min_unexpanded > worst_topk * ratio` AND a warm-up budget
/// has been used. This is a cheap "good enough" heuristic that ignores
/// distance distribution and uses only a multiplier.
#[derive(Debug, Clone)]
pub struct RatioTerminator {
    pub warmup: u64,
    pub ratio: f32,
    pub ef_max: u64,
}
impl RatioTerminator {
    pub fn new(warmup: usize, ratio: f32, ef_max: usize) -> Self {
        Self {
            warmup: warmup as u64,
            ratio,
            ef_max: ef_max as u64,
        }
    }
}
impl BeamTerminator for RatioTerminator {
    fn reset(&mut self) {}
    fn should_stop(
        &mut self,
        expansions: u64,
        min_unexpanded: f32,
        worst_topk: f32,
        _improved: bool,
    ) -> bool {
        if expansions >= self.ef_max {
            return true;
        }
        if expansions < self.warmup {
            return false;
        }
        worst_topk.is_finite() && min_unexpanded > worst_topk * self.ratio
    }
    fn ef_pool(&self) -> usize {
        self.ef_max as usize
    }
    fn name(&self) -> &'static str {
        "ratio"
    }
}

/// Online-quantile terminator.
///
/// Tracks the distribution of "expansion improvement deltas":
/// `(worst_topk_before - worst_topk_after).max(0.0)` per expansion. Once
/// `warmup` expansions have populated the estimator, stop when the
/// **current potential improvement** from `min_unexpanded` is below the
/// p-quantile of recent improvements, i.e. the next expansion is unlikely
/// to beat the typical improvement.
///
/// Mathematically: potential ≈ `(worst_topk - min_unexpanded).max(0.0)`.
/// If `potential < quantile(p)`, the expected return on continuing is
/// below the p-th percentile of historical gains — diminishing returns.
#[derive(Debug, Clone)]
pub struct QuantileTerminator {
    quantile: P2Quantile,
    pub warmup: u64,
    pub p: f64,
    pub ef_max: u64,
    last_worst_topk: f32,
    initialized: bool,
}
impl QuantileTerminator {
    pub fn new(warmup: usize, p: f64, ef_max: usize) -> Self {
        Self {
            quantile: P2Quantile::new(p),
            warmup: warmup as u64,
            p,
            ef_max: ef_max as u64,
            last_worst_topk: f32::INFINITY,
            initialized: false,
        }
    }
}
impl BeamTerminator for QuantileTerminator {
    fn reset(&mut self) {
        self.quantile = P2Quantile::new(self.p);
        self.last_worst_topk = f32::INFINITY;
        self.initialized = false;
    }
    fn should_stop(
        &mut self,
        expansions: u64,
        min_unexpanded: f32,
        worst_topk: f32,
        _improved: bool,
    ) -> bool {
        if expansions >= self.ef_max {
            return true;
        }

        // Record improvement delta from previous step.
        if self.initialized && worst_topk.is_finite() && self.last_worst_topk.is_finite() {
            let delta = (self.last_worst_topk - worst_topk).max(0.0) as f64;
            self.quantile.add(delta);
        }
        self.last_worst_topk = worst_topk;
        self.initialized = true;

        if expansions < self.warmup || self.quantile.count() < 5 {
            return false;
        }

        let potential = (worst_topk - min_unexpanded).max(0.0) as f64;
        let typical = self.quantile.quantile();
        potential < typical
    }
    fn ef_pool(&self) -> usize {
        self.ef_max as usize
    }
    fn name(&self) -> &'static str {
        "quantile"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_ef_stops_at_budget() {
        let mut t = FixedEfTerminator::new(8);
        assert!(!t.should_stop(7, 0.0, 0.0, false));
        assert!(t.should_stop(8, 0.0, 0.0, false));
    }

    #[test]
    fn ratio_respects_warmup() {
        let mut t = RatioTerminator::new(4, 1.2, 64);
        // During warmup we must keep expanding even if the heuristic
        // would otherwise say "stop".
        assert!(!t.should_stop(3, 10.0, 1.0, false));
        // After warmup, large gap triggers stop.
        assert!(t.should_stop(8, 10.0, 1.0, false));
    }

    #[test]
    fn quantile_respects_warmup_and_ef_max() {
        let mut t = QuantileTerminator::new(4, 0.5, 16);
        for i in 0..3 {
            assert!(!t.should_stop(i, 1.0, 2.0, false), "warmup step {i}");
        }
        assert!(t.should_stop(16, 1.0, 2.0, false), "ef_max enforced");
    }
}
