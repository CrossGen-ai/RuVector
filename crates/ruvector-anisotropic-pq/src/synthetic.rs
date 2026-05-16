//! Synthetic dataset generator.
//!
//! We generate a mixture-of-clusters embedding-like distribution: pick `c`
//! random unit-norm cluster centres in dimension `d`, then sample points
//! around them with Gaussian noise. Vectors are then L2-normalised so dot
//! product equals cosine similarity — the regime where anisotropic loss is
//! most relevant.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub struct Dataset {
    pub train: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub dim: usize,
}

pub fn make(n_train: usize, n_query: usize, dim: usize, n_clusters: usize, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0_f32, 1.0).unwrap();

    // Cluster centres.
    let mut centres: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            normalise(v)
        })
        .collect();
    // Spread the centres a bit so they're not collapsed by normalisation.
    for c in &mut centres {
        for x in c.iter_mut() { *x *= 3.0; }
    }

    let mut sample = |rng: &mut StdRng| {
        let ci = rng.gen_range(0..n_clusters);
        let centre = &centres[ci];
        let v: Vec<f32> = (0..dim).map(|j| centre[j] + 0.5 * normal.sample(rng)).collect();
        normalise(v)
    };

    let train: Vec<Vec<f32>> = (0..n_train).map(|_| sample(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_query).map(|_| sample(&mut rng)).collect();
    Dataset { train, queries, dim }
}

fn normalise(mut v: Vec<f32>) -> Vec<f32> {
    let mut n = 0.0f32;
    for x in &v { n += x * x; }
    let n = n.sqrt().max(1e-12);
    for x in &mut v { *x /= n; }
    v
}
