//! Synthetic clustered dataset for reproducible benchmarks.
//!
//! Generates `n` vectors of dimension `dim` drawn from `k_clusters`
//! Gaussian blobs with unit isotropic noise. This produces a realistic
//! kNN structure (real neighborhoods, non-trivial recall numbers)
//! while staying fully deterministic given a seed.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Clone)]
pub struct Dataset {
    pub vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub dim: usize,
}

impl Dataset {
    pub fn synthetic_clustered(
        n: usize,
        dim: usize,
        k_clusters: usize,
        n_queries: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Cluster centroids in [-5, 5]^d
        let centroids: Vec<Vec<f32>> = (0..k_clusters)
            .map(|_| (0..dim).map(|_| rng.gen_range(-5.0f32..5.0)).collect())
            .collect();

        let gen_point = |rng: &mut StdRng| -> Vec<f32> {
            let c = rng.gen_range(0..k_clusters);
            let mut v = centroids[c].clone();
            for x in v.iter_mut() {
                *x += rng.gen_range(-1.0f32..1.0);
            }
            v
        };

        let vectors: Vec<Vec<f32>> = (0..n).map(|_| gen_point(&mut rng)).collect();
        let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen_point(&mut rng)).collect();

        Self { vectors, queries, dim }
    }

    pub fn n(&self) -> usize {
        self.vectors.len()
    }
}
