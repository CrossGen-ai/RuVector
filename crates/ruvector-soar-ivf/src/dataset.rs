//! Deterministic synthetic dataset generation for SOAR-IVF benchmarks.
//!
//! Uses a self-contained LCG PRNG so runs are reproducible without any extra
//! dependency. The corpus is a mixture of `n_clusters` isotropic Gaussians with
//! shared variance, which mimics the "clumpy" structure real embedding
//! distributions display (and where partition-based indexes actually matter —
//! purely isotropic uniform data makes IVF differences invisible).
//!
//! Queries are drawn from the same distribution so ground truth is meaningful.

pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x51ed_beef_cafe_1234,
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard-normal sample via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_f64() + 1e-12).ln();
        let u2 = self.next_f64() * std::f64::consts::TAU;
        ((-2.0 * u1).sqrt() * u2.cos()) as f32
    }
    pub fn next_range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize % (hi - lo).max(1))
    }
}

#[derive(Debug, Clone)]
pub struct DatasetConfig {
    pub n_vectors: usize,
    pub dims: usize,
    pub n_queries: usize,
    pub n_clusters: usize,
    /// Standard deviation of within-cluster noise.
    pub sigma: f32,
    /// Standard deviation of cluster-center placement (larger = more separated).
    pub center_sigma: f32,
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            n_vectors: 20_000,
            dims: 64,
            n_queries: 500,
            n_clusters: 64,
            sigma: 1.0,
            center_sigma: 6.0,
            seed: 0x50A5_0000_ABCD_1234,
        }
    }
}

pub struct Dataset {
    pub config: DatasetConfig,
    pub vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
}

impl Dataset {
    /// Generate a Gaussian-mixture dataset. Uses separate RNG streams for
    /// centers / corpus / queries so datasets are stable across parameters.
    pub fn generate(config: DatasetConfig) -> Self {
        let mut crng = Lcg::new(config.seed);
        let centers: Vec<Vec<f32>> = (0..config.n_clusters)
            .map(|_| {
                (0..config.dims)
                    .map(|_| crng.next_normal() * config.center_sigma)
                    .collect()
            })
            .collect();

        let mut vrng = Lcg::new(config.seed.wrapping_add(0xAAAA_5555));
        let vectors: Vec<Vec<f32>> = (0..config.n_vectors)
            .map(|_| {
                let c = vrng.next_range(0, config.n_clusters);
                (0..config.dims)
                    .map(|d| centers[c][d] + vrng.next_normal() * config.sigma)
                    .collect()
            })
            .collect();

        let mut qrng = Lcg::new(config.seed.wrapping_add(0x7777_3333));
        let queries: Vec<Vec<f32>> = (0..config.n_queries)
            .map(|_| {
                let c = qrng.next_range(0, config.n_clusters);
                (0..config.dims)
                    .map(|d| centers[c][d] + qrng.next_normal() * config.sigma)
                    .collect()
            })
            .collect();

        Self {
            config,
            vectors,
            queries,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_is_deterministic() {
        let cfg = DatasetConfig {
            n_vectors: 200,
            dims: 8,
            n_queries: 10,
            n_clusters: 4,
            sigma: 1.0,
            center_sigma: 5.0,
            seed: 99,
        };
        let a = Dataset::generate(cfg.clone());
        let b = Dataset::generate(cfg);
        assert_eq!(a.vectors[0], b.vectors[0]);
        assert_eq!(a.queries[5], b.queries[5]);
    }

    #[test]
    fn dataset_shape() {
        let cfg = DatasetConfig {
            n_vectors: 100,
            dims: 12,
            n_queries: 7,
            n_clusters: 3,
            sigma: 1.0,
            center_sigma: 3.0,
            seed: 42,
        };
        let d = Dataset::generate(cfg);
        assert_eq!(d.vectors.len(), 100);
        assert_eq!(d.vectors[0].len(), 12);
        assert_eq!(d.queries.len(), 7);
        assert_eq!(d.queries[0].len(), 12);
    }
}
