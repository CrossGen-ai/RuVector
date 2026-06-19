//! Deterministic dataset generation for benchmarks and tests.
//!
//! Fixed-seed PRNG so every run is reproducible. No I/O, no entropy.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Generate N unit-normalized random vectors of dimension D.
pub fn random_unit_vectors(n: usize, dims: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).expect("valid normal");
    let mut out = Vec::with_capacity(n * dims);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..dims).map(|_| normal.sample(&mut rng)).collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in &mut v {
                *x /= norm;
            }
        }
        out.extend_from_slice(&v);
    }
    out
}

/// Clustered dataset: `n_clusters × n_per_cluster` Gaussian blobs on the unit
/// sphere. Returns `(flat_vectors, cluster_assignments)`.
pub fn clustered_unit_vectors(
    n_clusters: usize,
    n_per_cluster: usize,
    dims: usize,
    std_dev: f32,
    seed: u64,
) -> (Vec<f32>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal_c = Normal::new(0.0f32, 1.0).expect("normal");
    let noise = Normal::new(0.0f32, std_dev).expect("noise");

    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| normal_c.sample(&mut rng)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect();

    let n = n_clusters * n_per_cluster;
    let mut flat = Vec::with_capacity(n * dims);
    let mut assign = Vec::with_capacity(n);
    for (ci, center) in centers.iter().enumerate() {
        for _ in 0..n_per_cluster {
            let mut v: Vec<f32> =
                center.iter().map(|&c| c + noise.sample(&mut rng)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            flat.extend_from_slice(&v);
            assign.push(ci);
        }
    }
    (flat, assign)
}

/// Generate N queries near random cluster centers — ensures each query has
/// well-defined nearest neighbors.
pub fn clustered_queries(
    n: usize,
    dims: usize,
    dataset: &[f32],
    n_per_cluster: usize,
    std_dev: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0f32, std_dev * 0.5).expect("noise");
    let n_clusters = dataset.len() / dims / n_per_cluster;
    let pick = Uniform::new(0usize, n_clusters);

    (0..n)
        .map(|_| {
            let ci = pick.sample(&mut rng);
            let center = &dataset[ci * n_per_cluster * dims..(ci * n_per_cluster + 1) * dims];
            let mut v: Vec<f32> =
                center.iter().map(|&c| c + noise.sample(&mut rng)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

/// Brute-force ground truth: indices of k-NN per query.
pub fn ground_truth(
    dataset: &[f32],
    queries: &[Vec<f32>],
    dims: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    let n = dataset.len() / dims;
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(u32, f32)> = (0..n)
                .map(|i| {
                    let v = &dataset[i * dims..(i + 1) * dims];
                    let d: f32 = v.iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as u32, d)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            dists.truncate(k);
            dists.into_iter().map(|(idx, _)| idx).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_unit_normalized() {
        let data = random_unit_vectors(10, 8, 42);
        for c in data.chunks_exact(8) {
            let n: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn ground_truth_has_k_results() {
        let data = random_unit_vectors(50, 8, 1);
        let q = clustered_queries(5, 8, &data, 10, 0.1, 9);
        let gt = ground_truth(&data, &q, 8, 5);
        for nn in &gt {
            assert_eq!(nn.len(), 5);
        }
    }
}
