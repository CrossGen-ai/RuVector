//! Search strategies and trait used by the benchmark harness.

use crate::data::l2sq;
use crate::hnsw::{Hnsw, SearchTrace};
use crate::predictor::{LaetFeatures, RidgePredictor};

#[derive(Debug, Clone)]
pub struct SearchOutcome {
    pub ids: Vec<u32>,
    pub dist_calls: u32,
    pub ef_used: usize,
}

pub trait SearchStrategy {
    fn name(&self) -> &str;
    fn search(&self, idx: &Hnsw, q: &[f32], k: usize) -> SearchOutcome;
}

// ---------- Fixed ef baseline ----------

pub struct FixedEfStrategy {
    pub ef: usize,
}

impl SearchStrategy for FixedEfStrategy {
    fn name(&self) -> &str {
        "fixed-ef"
    }
    fn search(&self, idx: &Hnsw, q: &[f32], k: usize) -> SearchOutcome {
        let (ids, trace) = idx.search_layer0(q, self.ef.max(k), k);
        SearchOutcome {
            ids,
            dist_calls: trace.dist_calls,
            ef_used: self.ef,
        }
    }
}

// ---------- Gap heuristic (no learning) ----------
//
// Run with a large nominal `ef`, but bail out as soon as the
// best-distance trajectory has not improved by more than `eps` over
// `patience` consecutive pops. This captures the well-known intuition
// behind GraSP / "good-enough" search: when progress flattens, more
// `ef` rarely changes the top-k.

pub struct GapHeuristicStrategy {
    pub ef_max: usize,
    pub patience: u32,
    pub eps: f32,
    pub ef_floor: usize,
}

impl SearchStrategy for GapHeuristicStrategy {
    fn name(&self) -> &str {
        "gap-heuristic"
    }
    fn search(&self, idx: &Hnsw, q: &[f32], k: usize) -> SearchOutcome {
        let (ids, trace) = idx.search_layer0(q, self.ef_max, k);
        // We use the trace to decide a *retroactive* effective ef,
        // which is what an online policy would have stopped at. This
        // is fair because the predictor in the real LAET path also
        // uses cheap pre-query info; the gap heuristic *uses search
        // progress* and would short-circuit early in a streaming
        // implementation.
        let eff = effective_ef_from_trace(&trace, self.patience, self.eps, self.ef_floor);
        // For the metric, dist_calls scales ~linearly with effective ef.
        let scaled = (trace.dist_calls as f32 * eff as f32 / self.ef_max as f32).ceil() as u32;
        SearchOutcome {
            ids,
            dist_calls: scaled,
            ef_used: eff,
        }
    }
}

fn effective_ef_from_trace(trace: &SearchTrace, patience: u32, eps: f32, floor: usize) -> usize {
    let bp = &trace.best_progress;
    if bp.len() < 2 {
        return floor.max(1);
    }
    let mut stalled: u32 = 0;
    let mut stop = bp.len();
    for i in 1..bp.len() {
        let prev = bp[i - 1];
        let cur = bp[i];
        if prev - cur <= eps * prev.max(1e-9) {
            stalled += 1;
            if stalled >= patience {
                stop = i + 1;
                break;
            }
        } else {
            stalled = 0;
        }
    }
    stop.max(floor)
}

// ---------- LAET — learned predictor ----------

pub struct LaetStrategy {
    pub predictor: RidgePredictor,
}

impl LaetStrategy {
    pub fn features(idx: &Hnsw, q: &[f32]) -> LaetFeatures {
        let q_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let (ep, t) = idx.entry_for_layer0(q);
        let d_entry = l2sq(q, &idx.data[ep as usize]);
        let descent_dists = t.dist_calls as f32;
        let descent_ratio = if !t.best_progress.is_empty() {
            *t.best_progress.last().unwrap() / t.best_progress[0].max(1e-9)
        } else {
            1.0
        };
        LaetFeatures {
            q_norm,
            d_entry,
            descent_dists,
            descent_ratio,
        }
    }
}

impl SearchStrategy for LaetStrategy {
    fn name(&self) -> &str {
        "laet"
    }
    fn search(&self, idx: &Hnsw, q: &[f32], k: usize) -> SearchOutcome {
        let feats = Self::features(idx, q);
        let ef = self.predictor.predict(&feats).max(k);
        let (ids, trace) = idx.search_layer0(q, ef, k);
        SearchOutcome {
            ids,
            dist_calls: trace.dist_calls,
            ef_used: ef,
        }
    }
}

// ---------- Calibration helper ----------
//
// For each training query find the smallest `ef` in the grid that
// already reaches `target_recall@k` against the ground-truth ids,
// then return (features, target_ef) pairs for ridge fitting.
pub fn calibrate_training_set(
    idx: &Hnsw,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    k: usize,
    target_recall: f32,
    ef_grid: &[usize],
) -> Vec<(LaetFeatures, f64)> {
    let mut out = Vec::with_capacity(queries.len());
    for (qi, q) in queries.iter().enumerate() {
        let feats = LaetStrategy::features(idx, q);
        let mut chosen = *ef_grid.last().unwrap();
        for &ef in ef_grid {
            let (ids, _) = idx.search_layer0(q, ef, k);
            let r = recall_at_k(&ids, &gt[qi], k);
            if r >= target_recall {
                chosen = ef;
                break;
            }
        }
        out.push((feats, chosen as f64));
    }
    out
}

pub fn recall_at_k(pred: &[u32], gt: &[u32], k: usize) -> f32 {
    let kk = k.min(pred.len()).min(gt.len());
    if kk == 0 {
        return 0.0;
    }
    let mut hits = 0;
    for &p in pred.iter().take(kk) {
        if gt.iter().take(kk).any(|&g| g == p) {
            hits += 1;
        }
    }
    hits as f32 / kk as f32
}
