//! ruvector-laet — Learned Adaptive Early Termination for HNSW-like beam search.
//!
//! Provides three [`SearchStrategy`] implementations over a hermetic flat k-NN graph:
//!
//! * [`FixedEf`] — classic constant-ef baseline.
//! * [`PatienceStop`] — no-learning heuristic: stop when best-distance stagnates.
//! * [`LaetStop`] — learned linear predictor over 5 cheap traversal features.
//!
//! The learned model is trained by closed-form ridge regression against an oracle
//! label constructed from full FixedEf traversals — no external ML crate needed.

pub mod bench;
pub mod features;
pub mod graph;
pub mod train;

use std::collections::{BinaryHeap, HashSet};

use crate::features::Features;
use crate::graph::{l2sq, FlatGraph, Item, MinItem};
use crate::train::LinearModel;

pub trait SearchStrategy {
    /// Return `true` to STOP; `false` to continue.
    fn should_stop(&mut self, f: &Features) -> bool;
    /// Reset per-query internal state.
    fn reset(&mut self) {}
}

#[derive(Clone, Debug)]
pub struct FixedEf {
    pub ef: usize,
}
impl SearchStrategy for FixedEf {
    fn should_stop(&mut self, f: &Features) -> bool {
        // Standard HNSW: stop when candidate min exceeds result max — approximated
        // here by capping iterations at ef.
        (f.iter as usize) >= self.ef
    }
}

#[derive(Clone, Debug)]
pub struct PatienceStop {
    pub patience: usize,
    pub max_iter: usize,
}
impl SearchStrategy for PatienceStop {
    fn should_stop(&mut self, f: &Features) -> bool {
        (f.iters_since_improve as usize) >= self.patience || (f.iter as usize) >= self.max_iter
    }
}

#[derive(Clone, Debug)]
pub struct LaetStop {
    pub model: LinearModel,
    pub min_iter: usize,
    pub max_iter: usize,
}
impl SearchStrategy for LaetStop {
    fn should_stop(&mut self, f: &Features) -> bool {
        let it = f.iter as usize;
        if it < self.min_iter {
            return false;
        }
        if it >= self.max_iter {
            return true;
        }
        // Predictor returns "remaining improvement potential"; stop when it falls
        // below `threshold` (i.e. we believe further search won't help).
        self.model.score(f) < self.model.threshold
    }
}

pub struct Index {
    pub graph: FlatGraph,
    pub ef_min: usize,
}

impl Index {
    pub fn new(points: Vec<Vec<f32>>, m: usize) -> Self {
        Self { graph: FlatGraph::build(points, m), ef_min: 16 }
    }

    /// Beam search. Returns top-`k` ids and a per-iteration feature trace.
    /// `entry_ids` seed the beam; use one or more deterministically-picked ids.
    pub fn search<S: SearchStrategy>(
        &self,
        query: &[f32],
        k: usize,
        entry_ids: &[u32],
        strat: &mut S,
    ) -> (Vec<u32>, Vec<Features>) {
        strat.reset();
        let n = self.graph.len();
        let mut visited = HashSet::with_capacity(n / 4);
        // Min-heap of candidates to expand.
        let mut cands: BinaryHeap<MinItem> = BinaryHeap::new();
        // Max-heap of current top-k results.
        let mut results: BinaryHeap<Item> = BinaryHeap::new();

        for &eid in entry_ids {
            if visited.insert(eid) {
                let d = l2sq(query, &self.graph.points[eid as usize]);
                cands.push(MinItem(Item { dist: d, id: eid }));
                results.push(Item { dist: d, id: eid });
            }
        }

        let mut trace: Vec<Features> = Vec::with_capacity(64);
        // Seed best_prev from initial results so first-iter delta is 0, not INFINITY.
        let mut best_prev = results
            .iter()
            .map(|x| x.dist)
            .fold(f32::INFINITY, f32::min);
        let mut iters_since_improve = 0f32;

        let mut iter: usize = 0;
        while let Some(MinItem(cur)) = cands.pop() {
            // Get current best (min) distance in results (which is the WORST of the top-k
            // in a max-heap; we want the MIN, so pull from a temporary tracker).
            let best_now = results
                .iter()
                .map(|x| x.dist)
                .fold(f32::INFINITY, f32::min);
            let delta = (best_prev - best_now).max(0.0);
            if delta > 0.0 {
                iters_since_improve = 0.0;
            } else {
                iters_since_improve += 1.0;
            }
            best_prev = best_now;

            let f = Features {
                iter: iter as f32,
                best_dist: best_now,
                delta_best_dist: delta,
                iters_since_improve,
                ef_min_reached: (iter as f32) / (self.ef_min as f32),
            };
            trace.push(f);

            if iter >= self.ef_min && strat.should_stop(&f) {
                break;
            }

            // NB: real HNSW prunes candidates worse than worst-of-topk. We deliberately
            // omit that here so `ef` controls actual work — otherwise pruning trims
            // every strategy to the same footprint and the learned predictor has no
            // opportunity to save distance calls. LAET-style research targets the
            // wasted expansions performed by conservatively-tuned ef; keeping ef
            // "loose" reproduces that regime.

            // Expand neighbors.
            for &nb in &self.graph.neighbors[cur.id as usize] {
                if !visited.insert(nb) {
                    continue;
                }
                let d = l2sq(query, &self.graph.points[nb as usize]);
                cands.push(MinItem(Item { dist: d, id: nb }));
                if results.len() < k {
                    results.push(Item { dist: d, id: nb });
                } else if let Some(worst) = results.peek() {
                    if d < worst.dist {
                        results.pop();
                        results.push(Item { dist: d, id: nb });
                    }
                }
            }
            iter += 1;
        }

        let mut out: Vec<Item> = results.into_sorted_vec();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        (out.into_iter().take(k).map(|x| x.id).collect(), trace)
    }
}

/// Brute-force ground truth top-k.
pub fn ground_truth(points: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<Item> = BinaryHeap::with_capacity(k + 1);
    for (i, p) in points.iter().enumerate() {
        let d = l2sq(query, p);
        if heap.len() < k {
            heap.push(Item { dist: d, id: i as u32 });
        } else if let Some(t) = heap.peek() {
            if d < t.dist {
                heap.pop();
                heap.push(Item { dist: d, id: i as u32 });
            }
        }
    }
    let mut v: Vec<Item> = heap.into_sorted_vec();
    v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    v.into_iter().map(|x| x.id).collect()
}

pub fn recall_at_k(pred: &[u32], truth: &[u32]) -> f32 {
    let tset: HashSet<u32> = truth.iter().copied().collect();
    let hit: usize = pred.iter().filter(|i| tset.contains(i)).count();
    hit as f32 / truth.len() as f32
}

/// Build (features, label) training pairs by replaying FixedEf oracle searches and
/// labelling each iteration with "remaining useful improvement": how much best_dist
/// still drops between this iter and the end of the oracle traversal. Predictor
/// then learns to output a small value when almost no improvement remains.
pub fn build_training_set(
    index: &Index,
    train_queries: &[Vec<f32>],
    entry_ids: &[u32],
    oracle_ef: usize,
    k: usize,
) -> (Vec<[f32; Features::DIM]>, Vec<f32>) {
    let mut xs: Vec<[f32; Features::DIM]> = Vec::new();
    let mut ys: Vec<f32> = Vec::new();
    for q in train_queries {
        let mut oracle = FixedEf { ef: oracle_ef };
        let (_, trace) = index.search(q, k, entry_ids, &mut oracle);
        if trace.len() < 3 {
            continue;
        }
        // Find the "convergence iter": first iter where best_dist ≤ final_best × 1.01
        // (i.e. within 1% of final). Anything after convergence is wasted work.
        let final_best = trace.last().unwrap().best_dist;
        let target = final_best * 1.01 + 1e-6;
        let converge_iter = trace
            .iter()
            .position(|f| f.best_dist <= target)
            .unwrap_or(trace.len() - 1);
        for (i, f) in trace.iter().enumerate() {
            // Label = 1.0 while below convergence (keep searching), fades to 0
            // afterwards (stop). Ridge regression will learn a smooth transition.
            let label = if i < converge_iter { 1.0f32 } else { 0.0f32 };
            xs.push(f.as_array());
            ys.push(label);
        }
    }
    (xs, ys)
}
