//! IVF range search with centroid-ball pruning.
//!
//! Each posting list is annotated with the maximum distance from any of
//! its members to its centroid (`r_i`). Given a query with radius `r`,
//! a posting list can only hold answers if the centroid is at most
//! `sqrt(r2) + r_i` away from the query. All other posting lists are
//! skipped without touching their members. Within a probed posting list
//! we still exactly test each member to retain the exactness guarantee
//! that linear scan gives.

use crate::{sqr_l2, RangeHit, RangeIndex};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

pub struct IvfRange {
    centroids: Vec<Vec<f32>>,
    /// for each centroid: max squared distance from any assigned point.
    /// stored squared so the ball inequality stays in sqr space.
    radii2: Vec<f32>,
    /// for each centroid: points assigned to it, each (id, vector).
    postings: Vec<Vec<(u32, Vec<f32>)>>,
    pub probed_centroids_last: std::sync::atomic::AtomicUsize,
}

impl IvfRange {
    /// Train with a fixed number of centroids using a lightweight k-means++
    /// init and a handful of Lloyd iterations. This is deterministic.
    pub fn build_with(points: &[Vec<f32>], n_centroids: usize, iters: usize) -> Self {
        assert!(!points.is_empty());
        let dim = points[0].len();
        let n_centroids = n_centroids.min(points.len()).max(1);
        let mut rng = ChaCha20Rng::seed_from_u64(0x5EEDED);

        // k-means++ seed
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_centroids);
        centroids.push(points[rng.gen_range(0..points.len())].clone());
        while centroids.len() < n_centroids {
            let mut weights = Vec::with_capacity(points.len());
            let mut total = 0.0f64;
            for p in points {
                let mut best = f32::INFINITY;
                for c in &centroids {
                    let d = sqr_l2(p, c);
                    if d < best {
                        best = d;
                    }
                }
                let w = best as f64 + 1e-9;
                weights.push(w);
                total += w;
            }
            let mut target = rng.gen_range(0.0..total);
            let mut idx = 0usize;
            for (i, w) in weights.iter().enumerate() {
                target -= *w;
                if target <= 0.0 {
                    idx = i;
                    break;
                }
            }
            centroids.push(points[idx].clone());
        }

        // Lloyd iterations
        let mut assign = vec![0u32; points.len()];
        for _ in 0..iters {
            for (i, p) in points.iter().enumerate() {
                let mut best = f32::INFINITY;
                let mut best_c = 0u32;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = sqr_l2(p, c);
                    if d < best {
                        best = d;
                        best_c = ci as u32;
                    }
                }
                assign[i] = best_c;
            }
            let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; n_centroids];
            let mut counts = vec![0u32; n_centroids];
            for (i, p) in points.iter().enumerate() {
                let c = assign[i] as usize;
                counts[c] += 1;
                for j in 0..dim {
                    sums[c][j] += p[j];
                }
            }
            for c in 0..n_centroids {
                if counts[c] > 0 {
                    for j in 0..dim {
                        centroids[c][j] = sums[c][j] / counts[c] as f32;
                    }
                }
            }
        }

        // final assignment + posting lists + radii
        let mut postings: Vec<Vec<(u32, Vec<f32>)>> = vec![Vec::new(); n_centroids];
        let mut radii2 = vec![0.0f32; n_centroids];
        for (i, p) in points.iter().enumerate() {
            let mut best = f32::INFINITY;
            let mut best_c = 0u32;
            for (ci, c) in centroids.iter().enumerate() {
                let d = sqr_l2(p, c);
                if d < best {
                    best = d;
                    best_c = ci as u32;
                }
            }
            if best > radii2[best_c as usize] {
                radii2[best_c as usize] = best;
            }
            postings[best_c as usize].push((i as u32, p.clone()));
        }

        Self {
            centroids,
            radii2,
            postings,
            probed_centroids_last: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl RangeIndex for IvfRange {
    fn build(points: &[Vec<f32>]) -> Self {
        let n_centroids = ((points.len() as f32).sqrt() as usize).max(8).min(1024);
        Self::build_with(points, n_centroids, 6)
    }

    fn range_search(&self, query: &[f32], r2: f32) -> Vec<RangeHit> {
        let r = r2.sqrt();
        let mut out = Vec::new();
        let mut probed = 0usize;
        for (ci, c) in self.centroids.iter().enumerate() {
            let d_qc = sqr_l2(query, c).sqrt();
            let r_i = self.radii2[ci].sqrt();
            // ball-to-ball intersection: skip if centers are farther than r + r_i
            if d_qc > r + r_i {
                continue;
            }
            probed += 1;
            for (id, p) in &self.postings[ci] {
                let d = sqr_l2(query, p);
                if d <= r2 {
                    out.push((*id, d));
                }
            }
        }
        self.probed_centroids_last
            .store(probed, std::sync::atomic::Ordering::Relaxed);
        out
    }

    fn name(&self) -> &'static str {
        "ivf-range-ball-prune"
    }
}
