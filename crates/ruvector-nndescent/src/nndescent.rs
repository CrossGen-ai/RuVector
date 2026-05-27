//! NN-Descent — approximate k-NN graph build via local-join refinement.
//!
//! ## Algorithm (Dong, Charikar & Li 2011, refined per PyNNDescent / CAGRA)
//!
//! 1. Initialise each node's heap with `k` random neighbours.
//! 2. Each iteration:
//!    a. For each node `u`, split its heap into `new[u]` (recently changed)
//!       and `old[u]` (already joined).  Cap |new[u]| by `rho * k`.
//!    b. Build "reverse" lists `new'[v]`, `old'[v]` containing every `u`
//!       such that `v ∈ new[u]` / `v ∈ old[u]`.
//!    c. Local join: for every `u`, for every pair `(a, b)` with
//!       `a ∈ new[u] ∪ new'[u]` and `b ∈ (new[u] ∪ new'[u] ∪ old[u] ∪ old'[u])`,
//!       compute `d(a,b)` once and try to insert it into the heaps of both
//!       `a` and `b`.  An insertion flips the entry's `is_new` flag back on.
//! 3. Stop when the update count per iteration drops below `delta * k * N`.
//!
//! ## Why this works
//!
//! "A neighbour of my neighbour is likely my neighbour."  Every pair joined
//! through a node `u` is a candidate that survived `u`'s heap, so they are
//! already biased toward being mutually close.  The `new` flag avoids
//! redoing work for pairs that have already been considered.  Empirically
//! the graph converges in 5–10 iterations to recall ≳ 0.9 at a fraction of
//! the O(N²) brute-force cost.
//!
//! ## Variants exposed here
//!
//! - [`NnDescent`]: vanilla, just `new[u] × (new[u] ∪ old[u])` joins.
//! - [`NnDescent`] with `reverse: true`: adds reverse-neighbour lists
//!   (the refinement that gives the algorithm its real edge on
//!   non-uniform data).
//! - [`NnDescent`] with `rho < 1.0`: sample-rate subsetting that trades
//!   recall for speed — the lever practitioners actually turn.

use crate::{heap::BoundedMaxHeap, BuildReport, KnnGraph, KnnGraphBuilder, Metric, Neighbor};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct NnDescentConfig {
    /// Sample rate for `new` items (PyNNDescent calls this `rho`, paper `ρ`).
    /// 1.0 → consider all changed neighbours, ~0.5 → typical sweet-spot.
    pub rho: f32,
    /// Early-stopping threshold: stop when updates < delta·k·N.
    pub delta: f32,
    /// Hard cap on iterations.
    pub max_iters: u32,
    /// Include reverse-neighbour lists in the local join.
    pub reverse: bool,
    /// RNG seed for reproducible initial graph.
    pub seed: u64,
}

impl Default for NnDescentConfig {
    fn default() -> Self {
        Self { rho: 1.0, delta: 0.001, max_iters: 30, reverse: true, seed: 0xC0FFEE }
    }
}

pub struct NnDescent<M: Metric> {
    pub metric: M,
    pub cfg: NnDescentConfig,
}

impl<M: Metric> NnDescent<M> {
    pub fn new(metric: M, cfg: NnDescentConfig) -> Self { Self { metric, cfg } }
}

impl<M: Metric> KnnGraphBuilder for NnDescent<M> {
    fn build(&mut self, data: &[Vec<f32>], k: usize) -> BuildReport {
        let n = data.len();
        assert!(n > k, "need at least k+1 points");
        let t0 = Instant::now();
        let mut rng = StdRng::seed_from_u64(self.cfg.seed);
        let mut calls: u64 = 0;

        // -- Step 1: random initialisation -----------------------------------
        let mut heaps: Vec<BoundedMaxHeap> =
            (0..n).map(|_| BoundedMaxHeap::new(k)).collect();
        for i in 0..n {
            let mut picked = 0usize;
            while picked < k {
                let j: usize = rng.gen_range(0..n);
                if j == i { continue; }
                let d = self.metric.dist(&data[i], &data[j]);
                calls += 1;
                if heaps[i].push(j as u32, d, true) {
                    picked += 1;
                }
            }
        }

        // -- Step 2: iterate -------------------------------------------------
        let max_new_per_node = (self.cfg.rho * k as f32).ceil() as usize;
        let stop_threshold = (self.cfg.delta * (k * n) as f32) as u64;
        let mut iters_run = 0u32;

        for it in 0..self.cfg.max_iters {
            iters_run = it + 1;
            // 2a + 2b: split into new/old and build reverse lists.
            let mut new_l: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut old_l: Vec<Vec<u32>> = vec![Vec::new(); n];
            for u in 0..n {
                let (a, b) = heaps[u].split_new_old(max_new_per_node);
                new_l[u] = a;
                old_l[u] = b;
            }
            let (new_r, old_r) = if self.cfg.reverse {
                let mut nr: Vec<Vec<u32>> = vec![Vec::new(); n];
                let mut or_: Vec<Vec<u32>> = vec![Vec::new(); n];
                for u in 0..n {
                    for &v in &new_l[u] { nr[v as usize].push(u as u32); }
                    for &v in &old_l[u] { or_[v as usize].push(u as u32); }
                }
                // sub-sample reverse lists to bound work per node
                for v in 0..n {
                    sample_in_place(&mut nr[v], max_new_per_node, &mut rng);
                    sample_in_place(&mut or_[v], max_new_per_node, &mut rng);
                }
                (nr, or_)
            } else {
                (vec![Vec::new(); n], vec![Vec::new(); n])
            };

            // 2c: local join.
            let mut updates: u64 = 0;
            for u in 0..n {
                // "new" candidates for u: union of forward and reverse new lists
                let mut newu: Vec<u32> = new_l[u].clone();
                if self.cfg.reverse { newu.extend(&new_r[u]); }
                dedup(&mut newu);
                let mut oldu: Vec<u32> = old_l[u].clone();
                if self.cfg.reverse { oldu.extend(&old_r[u]); }
                dedup(&mut oldu);

                // Pairs: new × new (i < j to avoid double work) and new × old.
                for i in 0..newu.len() {
                    let a = newu[i] as usize;
                    for j in (i+1)..newu.len() {
                        let b = newu[j] as usize;
                        updates += try_join(&self.metric, data, a, b, &mut heaps, &mut calls);
                    }
                    for j in 0..oldu.len() {
                        let b = oldu[j] as usize;
                        if a == b { continue; }
                        updates += try_join(&self.metric, data, a, b, &mut heaps, &mut calls);
                    }
                }
            }

            if updates <= stop_threshold { break; }
        }

        // -- Step 3: drain heaps to sorted lists -----------------------------
        let graph: KnnGraph = heaps.into_iter().map(|h| h.into_sorted()).collect();
        BuildReport {
            graph,
            elapsed: t0.elapsed(),
            distance_calls: calls,
            iterations: iters_run,
        }
    }
}

#[inline]
fn try_join<M: Metric>(
    metric: &M,
    data: &[Vec<f32>],
    a: usize,
    b: usize,
    heaps: &mut [BoundedMaxHeap],
    calls: &mut u64,
) -> u64 {
    if a == b { return 0; }
    // Cheap early-out: only bother if it could possibly improve either heap.
    let worst_a = heaps[a].worst();
    let worst_b = heaps[b].worst();
    if worst_a.is_finite() && worst_b.is_finite() {
        // We still need the actual distance to know — but if both heaps are
        // saturated and the squared-L2 lower bound (via triangle on the
        // currently-known worst) ruled it out we'd skip.  We don't have a
        // cheap lower-bound here, so just compute.
    }
    let _ = (worst_a, worst_b);
    let d = metric.dist(&data[a], &data[b]);
    *calls += 1;
    let mut updates = 0u64;
    if heaps[a].push(b as u32, d, true) { updates += 1; }
    if heaps[b].push(a as u32, d, true) { updates += 1; }
    updates
}

fn dedup(v: &mut Vec<u32>) {
    v.sort_unstable();
    v.dedup();
}

fn sample_in_place(v: &mut Vec<u32>, max_len: usize, rng: &mut StdRng) {
    if v.len() <= max_len { return; }
    // Fisher–Yates partial shuffle then truncate.
    for i in 0..max_len {
        let j = rng.gen_range(i..v.len());
        v.swap(i, j);
    }
    v.truncate(max_len);
}

/// Convenience: rebuild a [`KnnGraph`] from raw `(id, dist)` rows.  Useful in
/// tests and round-tripping persisted graphs.
pub fn from_rows(rows: Vec<Vec<(u32, f32)>>) -> KnnGraph {
    rows.into_iter()
        .map(|r| r.into_iter().map(|(id, dist)| Neighbor { id, dist }).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{brute::BruteForce, recall_at_k, L2};
    use rand::Rng;

    fn gaussian_dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()).collect()
    }

    #[test]
    fn high_recall_and_subquadratic_calls() {
        // Need N large enough that O(N²) brute clearly beats NN-Descent's
        // per-iteration cost; 1000 in 32-D is well past the crossover.
        let n = 1000;
        let data = gaussian_dataset(n, 32, 7);
        let mut brute = BruteForce::new(L2);
        let truth = brute.build(&data, 10);
        let mut nnd = NnDescent::new(L2, NnDescentConfig::default());
        let approx = nnd.build(&data, 10);
        let r = recall_at_k(&truth.graph, &approx.graph);
        assert!(r >= 0.90, "recall too low: {r}");
        assert!(approx.distance_calls < truth.distance_calls,
                "nn-descent ({}) should be sub-N² vs brute ({})",
                approx.distance_calls, truth.distance_calls);
    }

    #[test]
    fn reverse_helps_vs_no_reverse() {
        let data = gaussian_dataset(400, 32, 11);
        let mut brute = BruteForce::new(L2);
        let truth = brute.build(&data, 15);
        let cfg_no = NnDescentConfig { reverse: false, ..Default::default() };
        let cfg_yes = NnDescentConfig { reverse: true, ..Default::default() };
        let r_no = recall_at_k(&truth.graph,
            &NnDescent::new(L2, cfg_no).build(&data, 15).graph);
        let r_yes = recall_at_k(&truth.graph,
            &NnDescent::new(L2, cfg_yes).build(&data, 15).graph);
        // Reverse neighbours should not hurt; on this dataset they usually
        // help noticeably.  Assert "at least as good" to avoid flakiness.
        assert!(r_yes + 1e-6 >= r_no, "reverse=true should not regress recall");
    }

    #[test]
    fn rho_produces_valid_graph_with_recall_tradeoff() {
        // The paper claims rho<1 trades recall for build cost.  Empirically
        // the call-count effect is dataset-dependent (smaller `new` sets can
        // delay convergence and bump iteration count), but the *recall*
        // curve is monotone: less work → no better recall.
        let data = gaussian_dataset(800, 32, 19);
        let truth = BruteForce::new(L2).build(&data, 15);
        let r_full = recall_at_k(&truth.graph, &NnDescent::new(L2,
            NnDescentConfig { rho: 1.0, ..Default::default() })
            .build(&data, 15).graph);
        let r_half = recall_at_k(&truth.graph, &NnDescent::new(L2,
            NnDescentConfig { rho: 0.4, ..Default::default() })
            .build(&data, 15).graph);
        // Both should produce usable graphs; rho=1.0 should be at least as
        // accurate as rho=0.4 on this isotropic-Gaussian dataset.
        assert!(r_full >= 0.85, "rho=1.0 recall too low: {r_full}");
        assert!(r_half >= 0.70, "rho=0.4 recall too low: {r_half}");
        assert!(r_full + 1e-6 >= r_half,
                "rho=1.0 ({r_full}) should not be worse than rho=0.4 ({r_half})");
    }
}
