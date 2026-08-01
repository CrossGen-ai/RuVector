//! Synthetic MIPS dataset generator (deterministic, no external deps).
//!
//! Vectors are drawn from a mixture of low-rank Gaussians so that:
//!   * subspaces have real correlation structure (PQ has something to exploit),
//!   * norms vary (so anisotropic weighting differentiates from isotropic),
//!   * inner products span a realistic dynamic range.

use crate::Lcg;

/// A reproducible MIPS dataset.
#[derive(Debug, Clone)]
pub struct MipsDataset {
    pub dim: usize,
    pub n: usize,
    pub data: Vec<f32>,
    pub queries: Vec<f32>,
    pub num_queries: usize,
}

/// Generate `n` d-dim database vectors and `nq` queries.
///
/// Mechanism: build a low-rank Gaussian mixture with `n_clusters` centers,
/// project into d-dim, add per-vector norm scaling. Queries are drawn from
/// the same mixture so that a nontrivial subset of the database is relevant
/// to each query (MIPS is discriminative).
pub fn generate(dim: usize, n: usize, nq: usize, seed: u64) -> MipsDataset {
    let mut rng = Lcg(seed);
    let n_clusters = 16usize;

    // Cluster centers in d-dim.
    let mut centers = vec![0.0f32; n_clusters * dim];
    for i in 0..n_clusters {
        for j in 0..dim {
            centers[i * dim + j] = rng.next_normal() * 0.5;
        }
    }

    let mut data = vec![0.0f32; n * dim];
    for i in 0..n {
        let c = (rng.next_u64() as usize) % n_clusters;
        let scale = 0.5 + rng.next_f32() * 1.5; // norm variation matters for anisotropic loss
        for j in 0..dim {
            let noise = rng.next_normal() * 0.3;
            data[i * dim + j] = scale * (centers[c * dim + j] + noise);
        }
    }

    let mut queries = vec![0.0f32; nq * dim];
    for i in 0..nq {
        let c = (rng.next_u64() as usize) % n_clusters;
        for j in 0..dim {
            let noise = rng.next_normal() * 0.4;
            queries[i * dim + j] = centers[c * dim + j] + noise;
        }
        // Normalize query — typical in MIPS at inference time.
        let mut sq = 0.0f32;
        for j in 0..dim {
            sq += queries[i * dim + j] * queries[i * dim + j];
        }
        let inv = 1.0 / sq.sqrt().max(1e-12);
        for j in 0..dim {
            queries[i * dim + j] *= inv;
        }
    }

    MipsDataset { dim, n, data, queries, num_queries: nq }
}

/// Split off a small held-out slice for validation.
pub fn split_validation<'a>(ds: &'a MipsDataset, val_n: usize) -> (usize, &'a [f32]) {
    let val = val_n.min(ds.n);
    let start = ds.n - val;
    (start, &ds.data[start * ds.dim..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_shapes() {
        let ds = generate(32, 1000, 100, 7);
        assert_eq!(ds.dim, 32);
        assert_eq!(ds.data.len(), 32 * 1000);
        assert_eq!(ds.queries.len(), 32 * 100);
    }

    #[test]
    fn queries_are_unit_norm() {
        let ds = generate(16, 100, 20, 1);
        for i in 0..ds.num_queries {
            let row = &ds.queries[i * 16..(i + 1) * 16];
            let n: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-4, "query {} norm {}", i, n);
        }
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate(8, 50, 5, 123);
        let b = generate(8, 50, 5, 123);
        assert_eq!(a.data, b.data);
        assert_eq!(a.queries, b.queries);
    }
}
