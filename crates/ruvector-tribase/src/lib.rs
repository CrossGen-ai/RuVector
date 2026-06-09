//! ruvector-tribase
//!
//! Triangle-inequality pruning for IVF nearest-neighbor search.
//!
//! Given an IVF index with centroids `c_j` and for each cluster member `x`
//! a precomputed `d(x, c_j)`, a query `q` satisfies:
//!
//! ```text
//! |d(q, c_j) - d(x, c_j)| <= d(q, x) <= d(q, c_j) + d(x, c_j)
//! ```
//!
//! If the current k-th best distance is `tau`, any `x` with
//! `|d(q, c_j) - d(x, c_j)| > tau` cannot improve the heap and can be
//! skipped without evaluating the full distance. Storing each posting
//! list sorted by `d(x, c_j)` lets us binary-search the admissible
//! window and visit only points inside it.
//!
//! Three indices are exposed for measurement:
//!   * [`FlatIndex`]      — exhaustive brute force baseline.
//!   * [`PlainIvfIndex`]  — IVF with no pruning.
//!   * [`TribaseIndex`]   — IVF with triangle-inequality window pruning.
//!
//! All three implement the [`AnnIndex`] trait, so they are swappable.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;

pub mod ivf;
pub mod tribase;

pub use ivf::{FlatIndex, PlainIvfIndex};
pub use tribase::TribaseIndex;

/// A scored neighbour returned by a search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub dist: f32,
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Stats captured by indices that want to expose pruning counts.
#[derive(Debug, Default, Clone, Copy)]
pub struct SearchStats {
    /// Total candidate points considered (after probe selection).
    pub considered: u64,
    /// Candidates pruned by the triangle inequality without a full
    /// distance computation.
    pub pruned: u64,
    /// Full Euclidean distances actually evaluated.
    pub full_dist: u64,
}

impl SearchStats {
    pub fn merge(&mut self, other: &SearchStats) {
        self.considered += other.considered;
        self.pruned += other.pruned;
        self.full_dist += other.full_dist;
    }
}

/// Index trait so all three backends are swappable.
pub trait AnnIndex {
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor>;
    fn search_with_stats(&self, q: &[f32], k: usize) -> (Vec<Neighbor>, SearchStats) {
        (self.search(q, k), SearchStats::default())
    }
    fn estimated_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}

/// Squared L2 distance — we work in squared distances throughout to
/// avoid an `sqrt` per candidate. Triangle inequalities still hold in
/// the actual L2 metric, so when we prune we compare actual distances
/// (sqrt of squared values) directly.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    sq_l2(a, b).sqrt()
}

/// k-means++ seeding + Lloyd iterations. Returns the centroid table
/// `n_clusters × dim`.
pub fn kmeans(
    data: &[Vec<f32>],
    n_clusters: usize,
    iters: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    assert!(!data.is_empty());
    let dim = data[0].len();
    let mut rng = StdRng::seed_from_u64(seed);

    // k-means++ init.
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);
    centroids.push(data[rng.gen_range(0..data.len())].clone());
    while centroids.len() < n_clusters {
        let mut d2: Vec<f32> = data
            .iter()
            .map(|x| {
                centroids
                    .iter()
                    .map(|c| sq_l2(x, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let sum: f32 = d2.iter().sum();
        if sum <= 0.0 {
            centroids.push(data[rng.gen_range(0..data.len())].clone());
            continue;
        }
        let mut u = rng.gen::<f32>() * sum;
        let mut pick = data.len() - 1;
        for (i, w) in d2.iter_mut().enumerate() {
            u -= *w;
            if u <= 0.0 {
                pick = i;
                break;
            }
        }
        centroids.push(data[pick].clone());
    }

    // Lloyd loop.
    for _ in 0..iters {
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; n_clusters];
        let mut counts: Vec<u32> = vec![0; n_clusters];
        for x in data {
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d = sq_l2(x, c);
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            for d in 0..dim {
                sums[best][d] += x[d];
            }
            counts[best] += 1;
        }
        for j in 0..n_clusters {
            if counts[j] == 0 {
                // Re-seed empty cluster from a random point.
                centroids[j] = data[rng.gen_range(0..data.len())].clone();
            } else {
                let inv = 1.0 / counts[j] as f32;
                for d in 0..dim {
                    centroids[j][d] = sums[j][d] * inv;
                }
            }
        }
    }

    centroids
}

/// Generate Gaussian clusters for deterministic benchmarking.
pub fn make_clustered(
    n: usize,
    dim: usize,
    n_centers: usize,
    spread: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_centers)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0f32)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_centers];
            (0..dim)
                .map(|d| c[d] + spread * (rng.gen::<f32>() - 0.5))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_orders_by_distance() {
        let mut v = vec![
            Neighbor { id: 1, dist: 0.9 },
            Neighbor { id: 2, dist: 0.1 },
            Neighbor { id: 3, dist: 0.5 },
        ];
        v.sort();
        assert_eq!(v[0].id, 2);
        assert_eq!(v[1].id, 3);
        assert_eq!(v[2].id, 1);
    }

    #[test]
    fn kmeans_recovers_clusters() {
        let data = make_clustered(2_000, 16, 8, 0.05, 42);
        let centroids = kmeans(&data, 8, 25, 1);
        assert_eq!(centroids.len(), 8);
        // Every centroid should have at least one nearby point.
        for c in &centroids {
            let min = data
                .iter()
                .map(|x| sq_l2(x, c))
                .fold(f32::INFINITY, f32::min);
            assert!(min < 1.0, "centroid too far from any point: {min}");
        }
    }

    #[test]
    fn sq_l2_matches_definition() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        assert!((sq_l2(&a, &b) - 25.0).abs() < 1e-6);
    }
}
