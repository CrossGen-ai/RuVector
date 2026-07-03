//! Pivot selection strategies.
//!
//! Pivot quality dominates Ptolemaic pruning performance: pairs of *far-apart*
//! pivots give the tightest bounds (numerator / d(p_a, p_b) favours large
//! denominators only when the numerator is proportionally larger — which
//! happens when the pivots are geometrically spread).
//!
//! Two strategies are provided:
//!
//! * [`select_random`] — cheap baseline. Samples pivots uniformly from the
//!   dataset.
//! * [`select_farthest_first`] — greedy Farthest-First Traversal (a.k.a. FFT
//!   or Gonzalez's 2-approximation for k-centre). Empirically the best
//!   general-purpose pivot selector for metric indexes.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::euclidean;

/// Sample `k` pivots uniformly at random from the dataset.
pub fn select_random(data: &[f32], dim: usize, k: usize, seed: u64) -> Vec<Vec<f32>> {
    let n = data.len() / dim;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        let idx = rng.gen_range(0..n);
        out.push(data[idx * dim..(idx + 1) * dim].to_vec());
    }
    out
}

/// Greedy Farthest-First Traversal pivot selection. First pivot is chosen
/// at random (seeded); each subsequent pivot maximises min-distance to the
/// already-chosen set. Total cost `O(k * n * dim)`.
pub fn select_farthest_first(data: &[f32], dim: usize, k: usize, seed: u64) -> Vec<Vec<f32>> {
    let n = data.len() / dim;
    let mut rng = StdRng::seed_from_u64(seed);
    let first = rng.gen_range(0..n);
    let mut chosen: Vec<Vec<f32>> = vec![data[first * dim..(first + 1) * dim].to_vec()];
    // running min-distance to chosen set
    let mut min_d = vec![f32::INFINITY; n];
    for i in 0..n {
        let d = euclidean(&chosen[0], &data[i * dim..(i + 1) * dim]);
        min_d[i] = d;
    }
    while chosen.len() < k {
        // pick the argmax min_d
        let mut best_i = 0usize;
        let mut best_v = -1.0f32;
        for i in 0..n {
            if min_d[i] > best_v {
                best_v = min_d[i];
                best_i = i;
            }
        }
        let pv = data[best_i * dim..(best_i + 1) * dim].to_vec();
        // refresh min_d against new pivot
        for i in 0..n {
            let d = euclidean(&pv, &data[i * dim..(i + 1) * dim]);
            if d < min_d[i] {
                min_d[i] = d;
            }
        }
        chosen.push(pv);
    }
    chosen
}
