//! Deterministic synthetic dataset utilities.
//!
//! We use Gaussian-cluster data rather than truly uniform random so
//! that early-termination has signal to exploit: real workloads also
//! exhibit clusterability.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct Dataset {
    pub dim: usize,
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
}

#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn randn(rng: &mut ChaCha8Rng) -> f32 {
    // Box-Muller. f32 precision is plenty for synthetic ANN data.
    let u1: f32 = rng.gen_range(1e-7f32..1.0);
    let u2: f32 = rng.gen_range(0.0f32..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Generate `n_base` points and `n_query` queries in `dim` dims
/// drawn from `n_clusters` isotropic Gaussian clusters with cluster
/// centers spread over `[-spread, spread]^dim`.
pub fn gen_clustered(
    seed: u64,
    n_base: usize,
    n_query: usize,
    dim: usize,
    n_clusters: usize,
    spread: f32,
    sigma: f32,
) -> Dataset {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            (0..dim)
                .map(|_| rng.gen_range(-spread..spread))
                .collect()
        })
        .collect();

    let sample = |rng: &mut ChaCha8Rng| -> Vec<f32> {
        let c = &centers[rng.gen_range(0..n_clusters)];
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            v.push(c[j] + sigma * randn(rng));
        }
        v
    };

    let base = (0..n_base).map(|_| sample(&mut rng)).collect();
    let queries = (0..n_query).map(|_| sample(&mut rng)).collect();
    Dataset { dim, base, queries }
}

/// Brute-force k-nearest neighbours by L2 — ground truth for recall.
pub fn brute_topk(base: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut all: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .map(|(i, b)| (i, l2sq(b, q)))
        .collect();
    all.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_seed() {
        let a = gen_clustered(42, 100, 10, 16, 4, 5.0, 1.0);
        let b = gen_clustered(42, 100, 10, 16, 4, 5.0, 1.0);
        assert_eq!(a.base, b.base);
        assert_eq!(a.queries, b.queries);
    }

    #[test]
    fn brute_topk_sorted() {
        let ds = gen_clustered(7, 200, 1, 8, 3, 3.0, 0.5);
        let r = brute_topk(&ds.base, &ds.queries[0], 5);
        for w in r.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }
}
