//! Recall and reconstruction-error metrics.

use crate::pq::{Code, Codebook, Vector};

/// Recall@k over the full query set. Both inputs are `n_queries × k` arrays
/// of database ids.
pub fn recall_at_k(pred: &[Vec<u32>], truth: &[Vec<u32>]) -> f32 {
    assert_eq!(pred.len(), truth.len());
    let mut hits = 0usize;
    let mut total = 0usize;
    for (p, t) in pred.iter().zip(truth) {
        let ts: std::collections::HashSet<u32> = t.iter().copied().collect();
        for id in p {
            if ts.contains(id) {
                hits += 1;
            }
        }
        total += t.len();
    }
    hits as f32 / total as f32
}

/// Mean squared reconstruction error across a set of vectors.
pub fn reconstruction_mse(codebook: &Codebook, xs: &[Vector], codes: &[Code]) -> f32 {
    let mut acc = 0.0_f64;
    let mut n = 0usize;
    for (x, c) in xs.iter().zip(codes) {
        let rec = codebook.reconstruct(c);
        for i in 0..x.len() {
            let e = x[i] - rec[i];
            acc += (e * e) as f64;
        }
        n += x.len();
    }
    (acc / n as f64) as f32
}
