//! Probe-budget strategies. Each one decides "stop now or keep scanning the
//! next inverted list" using only quantities already computed during the scan.
//!
//! - [`FixedNprobe`] — baseline, always continues until `max_nprobe` is hit.
//! - [`PlateauProbe`] — stop after `patience` consecutive probes that fail to
//!   improve the running best score. Heuristic: cheap, no extra distances.
//! - [`MarginBudget`] — stop when the *next* centroid's squared distance to
//!   the query exceeds the current k-th score by `margin`. This is a
//!   provable miss bound *only* when the cluster radius is zero; for real
//!   clusters `margin` absorbs the slack (set ≥ max-radius² for safety).

use crate::{ProbeState, ProbeStrategy, TopK};

/// Classical IVF: visit exactly `max_nprobe` lists. Used as the recall ceiling.
#[derive(Clone, Copy, Debug)]
pub struct FixedNprobe {
    pub nprobe: usize,
}

impl FixedNprobe {
    pub fn new(nprobe: usize) -> Self {
        Self { nprobe }
    }
}

impl ProbeStrategy for FixedNprobe {
    fn name(&self) -> &'static str {
        "fixed"
    }
    fn new_state(&self) -> ProbeState {
        ProbeState::default()
    }
    fn should_continue(
        &self,
        _state: &mut ProbeState,
        probes_done: usize,
        _nlist: usize,
        _next_centroid_sqd: Option<f32>,
        _topk: &TopK,
    ) -> bool {
        probes_done < self.nprobe
    }
}

/// Stop after `patience` probes that did not improve the best score.
///
/// Rationale: when the IVF lists are visited in centroid-distance order, the
/// best score typically drops sharply during the first few probes and then
/// plateaus once the true neighbors have been picked up. A short streak of
/// "no improvement" is strong evidence that further probes will not help.
#[derive(Clone, Copy, Debug)]
pub struct PlateauProbe {
    pub patience: usize,
}

impl PlateauProbe {
    pub fn new(patience: usize) -> Self {
        Self { patience }
    }
}

impl ProbeStrategy for PlateauProbe {
    fn name(&self) -> &'static str {
        "plateau"
    }
    fn new_state(&self) -> ProbeState {
        ProbeState {
            last_best: f32::INFINITY,
            stagnant: 0,
        }
    }
    fn should_continue(
        &self,
        state: &mut ProbeState,
        _probes_done: usize,
        _nlist: usize,
        _next_centroid_sqd: Option<f32>,
        topk: &TopK,
    ) -> bool {
        let best = topk.best();
        if best + f32::EPSILON < state.last_best {
            state.last_best = best;
            state.stagnant = 0;
        } else {
            state.stagnant += 1;
        }
        state.stagnant < self.patience
    }
}

/// Stop when the next centroid is too far to plausibly contain a top-k
/// candidate. Concretely: if `next_centroid_sqd > topk.kth() + margin`, no
/// vector in that list can rank ahead of the current k-th unless the cluster
/// radius² exceeds `margin`. For Gaussian clusters with std σ in each of `D`
/// dims, the expected radius² is `D · σ²`; set `margin` accordingly.
///
/// Setting `margin = 0` gives the strict lower-bound prune assuming
/// point-clusters (always safe in the degenerate case `radius=0`).
#[derive(Clone, Copy, Debug)]
pub struct MarginBudget {
    pub margin: f32,
    /// Minimum probes to scan before margin pruning is allowed to fire.
    pub warmup: usize,
}

impl MarginBudget {
    pub fn new(margin: f32, warmup: usize) -> Self {
        Self { margin, warmup }
    }
}

impl ProbeStrategy for MarginBudget {
    fn name(&self) -> &'static str {
        "margin"
    }
    fn new_state(&self) -> ProbeState {
        ProbeState::default()
    }
    fn should_continue(
        &self,
        _state: &mut ProbeState,
        probes_done: usize,
        _nlist: usize,
        next_centroid_sqd: Option<f32>,
        topk: &TopK,
    ) -> bool {
        if probes_done < self.warmup {
            return true;
        }
        let Some(nc) = next_centroid_sqd else {
            return false;
        };
        let kth = topk.kth();
        if !kth.is_finite() {
            return true;
        }
        // Continue iff next centroid is close enough relative to current kth.
        nc <= kth + self.margin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Id;

    fn make_full_topk(k: usize, kth: f32) -> TopK {
        let mut t = TopK::new(k);
        for i in 0..k {
            t.push(kth - 1.0 + i as f32 * 0.001, i as Id);
        }
        // best ≈ kth-1, worst ≈ kth-1+k*0.001
        t
    }

    #[test]
    fn fixed_runs_to_budget() {
        let s = FixedNprobe::new(5);
        let mut st = s.new_state();
        let tk = make_full_topk(3, 10.0);
        for i in 1..5 {
            assert!(s.should_continue(&mut st, i, 100, Some(0.0), &tk));
        }
        assert!(!s.should_continue(&mut st, 5, 100, Some(0.0), &tk));
    }

    #[test]
    fn plateau_triggers_after_patience() {
        let s = PlateauProbe::new(2);
        let mut st = s.new_state();
        // First three probes: best improves each time.
        let mut tk = TopK::new(3);
        tk.push(10.0, 0);
        assert!(s.should_continue(&mut st, 1, 100, Some(0.0), &tk));
        tk.push(5.0, 1);
        assert!(s.should_continue(&mut st, 2, 100, Some(0.0), &tk));
        tk.push(3.0, 2);
        assert!(s.should_continue(&mut st, 3, 100, Some(0.0), &tk));
        // Now stagnate: no better point arrives.
        assert!(s.should_continue(&mut st, 4, 100, Some(0.0), &tk)); // stagnant=1
        assert!(!s.should_continue(&mut st, 5, 100, Some(0.0), &tk)); // stagnant=2 -> stop
    }

    #[test]
    fn margin_prunes_when_centroid_is_far() {
        let s = MarginBudget::new(0.0, 1);
        let mut st = s.new_state();
        let tk = make_full_topk(3, 10.0);
        // Next centroid far past kth -> stop.
        assert!(!s.should_continue(&mut st, 2, 100, Some(100.0), &tk));
        // Next centroid within budget -> continue.
        assert!(s.should_continue(&mut st, 2, 100, Some(5.0), &tk));
    }

    #[test]
    fn margin_respects_warmup() {
        let s = MarginBudget::new(0.0, 4);
        let mut st = s.new_state();
        let tk = make_full_topk(3, 10.0);
        // Even a far centroid is allowed before warmup is reached.
        assert!(s.should_continue(&mut st, 1, 100, Some(1e9), &tk));
        assert!(s.should_continue(&mut st, 3, 100, Some(1e9), &tk));
        assert!(!s.should_continue(&mut st, 4, 100, Some(1e9), &tk));
    }
}
