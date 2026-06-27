//! Cheap per-query features for predicting the minimum `ef_search` required
//! to reach a target recall.
//!
//! The signal here is *Steiner-hardness*-style: queries that fall in low-
//! density regions, far from the index manifold, or that land on a high-
//! degree entry node need a larger beam to escape local minima. We avoid any
//! feature that itself requires running an HNSW search — feature extraction
//! cost must stay well below the cost of a small-`ef` search.
//!
//! Features (FEATURE_DIM = 8):
//!   0 - bias term (1.0)
//!   1 - query L2 norm
//!   2 - per-dim mean(|x|)
//!   3 - per-dim std
//!   4 - sq_l2 distance to nearest medoid
//!   5 - sq_l2 distance to mean of medoids
//!   6 - log(out-degree at layer-0 entry point + 1)
//!   7 - sq_l2 to entry-point vector

use crate::hnsw::{sq_l2, Hnsw, SearchStats};

pub const FEATURE_DIM: usize = 8;

#[derive(Debug, Clone)]
pub struct QueryFeatures {
    pub x: [f32; FEATURE_DIM],
}

/// A small set of medoids learned by k-means++ over a sample of the data.
/// 16 medoids is enough to give the predictor a coarse density signal.
#[derive(Debug, Clone)]
pub struct Medoids {
    pub dim: usize,
    pub centers: Vec<f32>, // [k * dim]
    pub mean: Vec<f32>,    // [dim]
}

impl Medoids {
    /// Pick `k` medoids using a deterministic k-means++ seed walk from `data`.
    pub fn fit(data: &[f32], dim: usize, k: usize, seed: u64) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let n = data.len() / dim;
        assert!(n >= k);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut chosen: Vec<usize> = Vec::with_capacity(k);
        chosen.push(rng.gen_range(0..n));
        let mut d2: Vec<f32> = (0..n)
            .map(|i| sq_l2(&data[i * dim..(i + 1) * dim], &data[chosen[0] * dim..(chosen[0] + 1) * dim]))
            .collect();
        for _ in 1..k {
            let total: f32 = d2.iter().sum();
            let mut r = rng.gen_range(0.0_f32..total.max(1e-9));
            let mut pick = 0usize;
            for (i, &dv) in d2.iter().enumerate() {
                r -= dv;
                if r <= 0.0 {
                    pick = i;
                    break;
                }
            }
            chosen.push(pick);
            for i in 0..n {
                let nd = sq_l2(
                    &data[i * dim..(i + 1) * dim],
                    &data[pick * dim..(pick + 1) * dim],
                );
                if nd < d2[i] {
                    d2[i] = nd;
                }
            }
        }
        let mut centers = vec![0.0_f32; k * dim];
        for (ci, &idx) in chosen.iter().enumerate() {
            centers[ci * dim..(ci + 1) * dim].copy_from_slice(&data[idx * dim..(idx + 1) * dim]);
        }
        let mut mean = vec![0.0_f32; dim];
        for c in 0..k {
            for d in 0..dim {
                mean[d] += centers[c * dim + d];
            }
        }
        for d in 0..dim {
            mean[d] /= k as f32;
        }
        Self { dim, centers, mean }
    }

    pub fn k(&self) -> usize {
        self.centers.len() / self.dim
    }

    fn nearest_dist(&self, q: &[f32]) -> f32 {
        let mut best = f32::INFINITY;
        for c in 0..self.k() {
            let d = sq_l2(&self.centers[c * self.dim..(c + 1) * self.dim], q);
            if d < best {
                best = d;
            }
        }
        best
    }
}

/// Extract features for a query. The entry-point lookup walks the HNSW upper
/// layers (cheap — O(M log N) distance computations) and the cost is counted
/// in `stats` so that all overheads are visible in the final benchmark.
pub fn extract_features(
    q: &[f32],
    medoids: &Medoids,
    index: &Hnsw,
    stats: &mut SearchStats,
) -> QueryFeatures {
    let dim = q.len();
    let mut x = [0.0_f32; FEATURE_DIM];
    x[0] = 1.0;
    let mut norm = 0.0_f32;
    let mut s_abs = 0.0_f32;
    let mut m1 = 0.0_f32;
    let mut m2 = 0.0_f32;
    for &v in q {
        norm += v * v;
        s_abs += v.abs();
        m1 += v;
        m2 += v * v;
    }
    let n = dim as f32;
    let mean = m1 / n;
    let var = (m2 / n) - mean * mean;
    x[1] = norm.sqrt();
    x[2] = s_abs / n;
    x[3] = var.max(0.0).sqrt();
    x[4] = medoids.nearest_dist(q);
    x[5] = sq_l2(&medoids.mean, q);
    if let Some((ep, ep_d)) = index.descend_to_layer0(q, stats) {
        x[6] = ((index.out_degree0(ep) as f32) + 1.0).ln();
        x[7] = ep_d;
    } else {
        x[6] = 0.0;
        x[7] = 0.0;
    }
    QueryFeatures { x }
}
