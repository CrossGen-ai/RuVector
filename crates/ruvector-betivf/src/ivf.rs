//! Plain IVF index: spherical-agnostic k-means (Lloyd) partitioning with per-cluster
//! `radius` = max member-to-centroid L2 distance, used by BET search for sound
//! lower bounds via the triangle inequality.

use crate::{l2, l2_sq, BetIvfError, Result};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Partition {
    pub centroid: Vec<f32>,
    pub radius: f32,
    pub members: Vec<u32>,
}

#[derive(Debug)]
pub struct IvfIndex {
    pub dim: usize,
    pub vectors: Vec<f32>,
    pub n: usize,
    pub partitions: Vec<Partition>,
}

impl IvfIndex {
    pub fn build<R: Rng>(
        dim: usize,
        vectors: Vec<f32>,
        n_clusters: usize,
        n_iters: usize,
        rng: &mut R,
    ) -> Result<Self> {
        if vectors.is_empty() {
            return Err(BetIvfError::EmptyDataset);
        }
        if vectors.len() % dim != 0 {
            return Err(BetIvfError::InvalidParameter(format!(
                "vectors len {} not multiple of dim {}",
                vectors.len(),
                dim
            )));
        }
        let n = vectors.len() / dim;
        if n_clusters == 0 || n_clusters > n {
            return Err(BetIvfError::InvalidParameter(format!(
                "n_clusters {n_clusters} must be in 1..={n}"
            )));
        }

        let mut centroids = init_kmeans_pp(dim, &vectors, n, n_clusters, rng);
        let mut assign = vec![0u32; n];
        for _ in 0..n_iters {
            for i in 0..n {
                let v = &vectors[i * dim..(i + 1) * dim];
                let mut best = 0u32;
                let mut best_d = f32::INFINITY;
                for (c, cent) in centroids.iter().enumerate() {
                    let d = l2_sq(v, cent);
                    if d < best_d {
                        best_d = d;
                        best = c as u32;
                    }
                }
                assign[i] = best;
            }
            let mut sums = vec![0.0f32; n_clusters * dim];
            let mut counts = vec![0u32; n_clusters];
            for i in 0..n {
                let v = &vectors[i * dim..(i + 1) * dim];
                let c = assign[i] as usize;
                counts[c] += 1;
                let s = &mut sums[c * dim..(c + 1) * dim];
                for j in 0..dim {
                    s[j] += v[j];
                }
            }
            for c in 0..n_clusters {
                if counts[c] == 0 {
                    let i = rng.gen_range(0..n);
                    centroids[c] =
                        vectors[i * dim..(i + 1) * dim].to_vec();
                } else {
                    let inv = 1.0 / counts[c] as f32;
                    let cent = &mut centroids[c];
                    let s = &sums[c * dim..(c + 1) * dim];
                    for j in 0..dim {
                        cent[j] = s[j] * inv;
                    }
                }
            }
        }

        let mut partitions: Vec<Partition> = centroids
            .into_iter()
            .map(|c| Partition {
                centroid: c,
                radius: 0.0,
                members: Vec::new(),
            })
            .collect();
        for i in 0..n {
            let c = assign[i] as usize;
            partitions[c].members.push(i as u32);
        }
        for p in &mut partitions {
            let mut r = 0.0f32;
            for &m in &p.members {
                let v = &vectors[m as usize * dim..(m as usize + 1) * dim];
                let d = l2(v, &p.centroid);
                if d > r {
                    r = d;
                }
            }
            p.radius = r;
        }

        Ok(Self {
            dim,
            vectors,
            n,
            partitions,
        })
    }

    #[inline]
    pub fn vector(&self, id: u32) -> &[f32] {
        let i = id as usize;
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }
}

fn init_kmeans_pp<R: Rng>(
    dim: usize,
    vectors: &[f32],
    n: usize,
    k: usize,
    rng: &mut R,
) -> Vec<Vec<f32>> {
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    let first = rng.gen_range(0..n);
    centroids.push(vectors[first * dim..(first + 1) * dim].to_vec());
    let mut dists = vec![f32::INFINITY; n];
    for _ in 1..k {
        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let d = l2_sq(v, centroids.last().unwrap());
            if d < dists[i] {
                dists[i] = d;
            }
        }
        let total: f64 = dists.iter().map(|&d| d as f64).sum();
        if total <= 0.0 {
            let i = rng.gen_range(0..n);
            centroids.push(vectors[i * dim..(i + 1) * dim].to_vec());
            continue;
        }
        let mut target = rng.gen::<f64>() * total;
        let mut chosen = 0usize;
        for i in 0..n {
            target -= dists[i] as f64;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen * dim..(chosen + 1) * dim].to_vec());
    }
    centroids
}
