//! Brute-force float L2 baseline (ground truth oracle).

use crate::sq_l2;

/// Return top-k (idx, sq_dist) pairs sorted ascending by distance.
pub fn flat_l2_topk(db: &[f32], n: usize, d: usize, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    assert_eq!(db.len(), n * d);
    assert_eq!(query.len(), d);
    let mut all: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, sq_l2(&db[i * d..(i + 1) * d], query)))
        .collect();
    all.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all
}
