//! Synthetic dataset generation for reproducible benchmarks.
//!
//! We construct a mixture-of-Gaussians dataset with a controlled fraction of
//! "hub" points — high-norm vectors that appear near many queries but whose
//! own k-NN neighborhood is dominated by other hubs. This is precisely the
//! failure mode Reverse-KNN Verified Retrieval is designed to filter.

use crate::FlatL2Index;

/// Deterministic 64-bit SplitMix generator.
#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Standard normal via Box–Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_f32()).max(1e-9);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

pub struct GenSpec {
    pub n: usize,
    pub dim: usize,
    pub n_clusters: usize,
    pub hub_frac: f32,
    pub hub_scale: f32,
    pub seed: u64,
}

impl Default for GenSpec {
    fn default() -> Self {
        Self {
            n: 5_000,
            dim: 64,
            n_clusters: 16,
            hub_frac: 0.05,
            hub_scale: 3.5,
            seed: 0xC0FFEE,
        }
    }
}

pub struct Generated {
    pub index: FlatL2Index,
    pub queries: Vec<Vec<f32>>,
    pub hub_ids: Vec<usize>,
}

/// Build a mixture-of-Gaussians dataset + a disjoint set of queries drawn
/// from the same mixture. Hub points get their norm inflated by `hub_scale`
/// so they collect asymmetric neighborhoods.
pub fn generate(spec: &GenSpec, n_queries: usize) -> Generated {
    let mut rng = SplitMix64::new(spec.seed);

    // Cluster centers ~ N(0, 4 I).
    let mut centers: Vec<Vec<f32>> = Vec::with_capacity(spec.n_clusters);
    for _ in 0..spec.n_clusters {
        let mut c = Vec::with_capacity(spec.dim);
        for _ in 0..spec.dim {
            c.push(2.0 * rng.next_normal());
        }
        centers.push(c);
    }

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(spec.n);
    let mut hub_ids: Vec<usize> = Vec::new();
    for i in 0..spec.n {
        let ci = (rng.next_u64() as usize) % spec.n_clusters;
        let c = &centers[ci];
        let mut v = Vec::with_capacity(spec.dim);
        for d in 0..spec.dim {
            v.push(c[d] + rng.next_normal());
        }
        let is_hub = rng.next_f32() < spec.hub_frac;
        if is_hub {
            for d in 0..spec.dim {
                v[d] *= spec.hub_scale;
            }
            hub_ids.push(i);
        }
        rows.push(v);
    }
    let index = FlatL2Index::from_rows(spec.dim, &rows);

    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let ci = (rng.next_u64() as usize) % spec.n_clusters;
        let c = &centers[ci];
        let mut v = Vec::with_capacity(spec.dim);
        for d in 0..spec.dim {
            v.push(c[d] + rng.next_normal());
        }
        queries.push(v);
    }

    Generated {
        index,
        queries,
        hub_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NnIndex;

    #[test]
    fn generate_is_reproducible() {
        let spec = GenSpec {
            n: 200,
            dim: 8,
            n_clusters: 4,
            hub_frac: 0.1,
            hub_scale: 3.0,
            seed: 7,
        };
        let a = generate(&spec, 5);
        let b = generate(&spec, 5);
        assert_eq!(a.queries.len(), b.queries.len());
        assert_eq!(a.hub_ids, b.hub_ids);
        for i in 0..5 {
            assert_eq!(a.queries[i], b.queries[i]);
        }
    }

    #[test]
    fn hubs_have_higher_norm_than_average() {
        let spec = GenSpec::default();
        let gen = generate(&spec, 10);
        if gen.hub_ids.is_empty() {
            return;
        }
        let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mean_all: f32 = (0..spec.n)
            .map(|i| norm(gen.index.vector(i)))
            .sum::<f32>()
            / spec.n as f32;
        let mean_hub: f32 = gen
            .hub_ids
            .iter()
            .map(|&i| norm(gen.index.vector(i)))
            .sum::<f32>()
            / gen.hub_ids.len() as f32;
        assert!(mean_hub > mean_all * 1.5, "hub norm {mean_hub} vs all {mean_all}");
    }
}
