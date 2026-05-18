//! Synthetic OOD dataset generator.
//!
//! Base vectors are drawn from a Gaussian mixture model (GMM) with `n_clusters`
//! cluster centres in `dim` dimensions.  Query vectors come from a *shifted*
//! GMM whose cluster centres are offset from the base centres by a fixed amount,
//! modelling the cross-modal OOD scenario (e.g. CLIP text → image embeddings).
//!
//! Ground-truth k-NN is computed via brute-force exact search so recall can
//! be measured precisely.

use std::collections::HashSet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::graph::l2sq;

/// Parameters for the synthetic OOD dataset.
#[derive(Debug, Clone)]
pub struct DatasetParams {
    /// Number of base vectors.
    pub n_base: usize,
    /// Dimensionality.
    pub dim: usize,
    /// Number of Gaussian clusters in the base GMM.
    pub n_clusters: usize,
    /// Spread (std-dev) of each cluster.
    pub cluster_std: f32,
    /// Mean of each cluster's coordinate (drawn uniformly from [-range, range]).
    pub cluster_range: f32,
    /// Shift applied to query cluster centres relative to base cluster centres.
    pub ood_shift: f32,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for DatasetParams {
    fn default() -> Self {
        DatasetParams {
            n_base: 5_000,
            dim: 64,
            n_clusters: 8,
            cluster_std: 0.5,
            cluster_range: 5.0,
            ood_shift: 3.0,
            seed: 42,
        }
    }
}

/// A generated OOD dataset.
pub struct OodDataset {
    /// Base corpus vectors.
    pub base: Vec<Vec<f32>>,
    /// Training queries (from shifted distribution, used for index construction).
    pub train_queries: Vec<Vec<f32>>,
    /// Test queries (from shifted distribution, used for recall evaluation).
    pub test_queries: Vec<Vec<f32>>,
    /// Ground-truth k-NN sets for each test query (indices into `base`).
    pub ground_truth: Vec<HashSet<usize>>,
}

/// Generate a synthetic OOD dataset according to `params`.
pub fn generate_ood_dataset(
    params: &DatasetParams,
    n_train_queries: usize,
    n_test_queries: usize,
    k_gt: usize,
) -> OodDataset {
    let mut rng = StdRng::seed_from_u64(params.seed);

    // Sample base cluster centres
    let base_centers: Vec<Vec<f32>> = (0..params.n_clusters)
        .map(|_| {
            (0..params.dim)
                .map(|_| rng.gen_range(-params.cluster_range..params.cluster_range))
                .collect()
        })
        .collect();

    // Query cluster centres = base centres + constant shift on first half of dims
    let query_centers: Vec<Vec<f32>> = base_centers
        .iter()
        .map(|c| {
            c.iter()
                .enumerate()
                .map(|(d, &x)| if d < params.dim / 2 { x + params.ood_shift } else { x })
                .collect()
        })
        .collect();

    // Generate base vectors
    let base = sample_gmm(&base_centers, params.n_base, params.cluster_std, &mut rng);

    // Generate train and test query vectors from the shifted GMM
    let train_queries = sample_gmm(&query_centers, n_train_queries, params.cluster_std, &mut rng);
    let test_queries = sample_gmm(&query_centers, n_test_queries, params.cluster_std, &mut rng);

    // Brute-force ground truth for test queries
    let ground_truth: Vec<HashSet<usize>> = test_queries
        .iter()
        .map(|q| exact_knn(q, &base, k_gt))
        .collect();

    OodDataset {
        base,
        train_queries,
        test_queries,
        ground_truth,
    }
}

/// Sample `n` vectors from a GMM (uniform cluster assignment + Gaussian noise).
fn sample_gmm(
    centers: &[Vec<f32>],
    n: usize,
    std: f32,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let dim = centers[0].len();
    let nc = centers.len();
    (0..n)
        .map(|i| {
            let c = &centers[i % nc];
            c.iter()
                .map(|&x| x + std * gaussian_noise(rng))
                .collect()
        })
        .collect()
}

/// Box-Muller transform for a standard normal sample.
fn gaussian_noise(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen_range(1e-7..1.0);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Exact brute-force k-NN of `query` in `corpus`.
pub fn exact_knn(query: &[f32], corpus: &[Vec<f32>], k: usize) -> HashSet<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2sq(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(k).map(|(id, _)| *id).collect()
}

/// Compute recall@k: fraction of true neighbours found in `results`.
pub fn recall_at_k(result_ids: &[usize], ground_truth: &HashSet<usize>) -> f64 {
    let hits = result_ids.iter().filter(|id| ground_truth.contains(id)).count();
    hits as f64 / ground_truth.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_sizes() {
        let params = DatasetParams {
            n_base: 200,
            dim: 16,
            n_clusters: 4,
            cluster_std: 0.5,
            cluster_range: 3.0,
            ood_shift: 2.0,
            seed: 99,
        };
        let ds = generate_ood_dataset(&params, 50, 30, 5);
        assert_eq!(ds.base.len(), 200);
        assert_eq!(ds.train_queries.len(), 50);
        assert_eq!(ds.test_queries.len(), 30);
        assert_eq!(ds.ground_truth.len(), 30);
    }

    #[test]
    fn test_exact_knn_correctness() {
        // 3 vectors; query = first; nearest should be self (id=0)
        let corpus = vec![
            vec![0.0f32, 0.0],
            vec![10.0, 10.0],
            vec![20.0, 20.0],
        ];
        let gt = exact_knn(&[0.0, 0.0], &corpus, 1);
        assert!(gt.contains(&0));
    }

    #[test]
    fn test_recall_at_k() {
        let gt: HashSet<usize> = [0, 1, 2].iter().cloned().collect();
        assert!((recall_at_k(&[0, 1, 2], &gt) - 1.0).abs() < 1e-9);
        assert!((recall_at_k(&[0, 5, 6], &gt) - 1.0 / 3.0).abs() < 1e-9);
    }
}
