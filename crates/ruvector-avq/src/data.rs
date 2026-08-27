//! Deterministic synthetic corpus: mixture-of-Gaussians in `dim`
//! dimensions with per-cluster mean and covariance drawn from a
//! seeded RNG. Vectors are NOT unit-normalised so norm variance
//! matters — that is precisely what AvqNorm targets.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

pub struct Corpus {
    pub train: Vec<Vec<f32>>,
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub dim: usize,
}

pub fn synth_corpus(
    dim: usize,
    n_base: usize,
    n_queries: usize,
    n_train: usize,
    n_clusters: usize,
    seed: u64,
) -> Corpus {
    let mut rng = StdRng::seed_from_u64(seed);
    // Cluster means uniformly on [-1, 1]^dim.
    let means: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    // Per-cluster diagonal std (heteroskedastic — some clusters wider).
    let stds: Vec<f32> = (0..n_clusters)
        .map(|_| rng.gen_range(0.08..0.35))
        .collect();
    // Per-cluster norm scale — pushes norms apart so AvqNorm has work
    // to do.
    let scales: Vec<f32> = (0..n_clusters)
        .map(|_| rng.gen_range(0.4..2.5))
        .collect();

    let mut sample = |rng: &mut StdRng| -> Vec<f32> {
        let c = rng.gen_range(0..n_clusters);
        let mean = &means[c];
        let std = stds[c];
        let scale = scales[c];
        let n = Normal::new(0.0f32, std).unwrap();
        mean.iter()
            .map(|m| (m + n.sample(rng)) * scale)
            .collect()
    };

    let base = (0..n_base).map(|_| sample(&mut rng)).collect();
    let queries = (0..n_queries).map(|_| sample(&mut rng)).collect();
    let train = (0..n_train).map(|_| sample(&mut rng)).collect();
    Corpus { train, base, queries, dim }
}
