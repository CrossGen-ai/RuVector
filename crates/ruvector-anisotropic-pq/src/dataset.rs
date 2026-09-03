//! Deterministic synthetic dataset with clustered structure that gives
//! anisotropic quantization something to exploit.
//!
//! We draw `n` points from a Gaussian mixture with `k_clusters` isotropic
//! components in `dim` dimensions. Cluster centres are drawn from a
//! zero-mean unit-variance Gaussian and rescaled to unit norm; per-point
//! noise is isotropic Gaussian at `noise_std`.
//!
//! The RNG is a simple splittable linear-congruential generator so results
//! are byte-identical across machines with no external crate dependency.

use crate::pq::Vector;

#[derive(Debug, Clone, Copy)]
pub struct DatasetConfig {
    pub dim: usize,
    pub n_db: usize,
    pub n_queries: usize,
    pub k_clusters: usize,
    pub noise_std: f32,
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            n_db: 5_000,
            n_queries: 500,
            k_clusters: 16,
            noise_std: 0.35,
            seed: 0xA155_0741,
        }
    }
}

pub struct GaussianMixture {
    pub db: Vec<Vector>,
    pub queries: Vec<Vector>,
    pub cfg: DatasetConfig,
}

impl GaussianMixture {
    pub fn generate(cfg: DatasetConfig) -> Self {
        let mut rng = SplitMix64::new(cfg.seed);
        let mut centres: Vec<Vector> = (0..cfg.k_clusters)
            .map(|_| rand_unit_vector(&mut rng, cfg.dim))
            .collect();
        // Keep centre norms modest but variable so MIPS ranking is meaningful.
        for (i, c) in centres.iter_mut().enumerate() {
            let scale = 1.0 + 0.25 * (i as f32);
            for v in c.iter_mut() {
                *v *= scale;
            }
        }

        let mut db = Vec::with_capacity(cfg.n_db);
        for i in 0..cfg.n_db {
            let cid = i % cfg.k_clusters;
            db.push(perturb(&centres[cid], cfg.noise_std, &mut rng));
        }

        let mut queries = Vec::with_capacity(cfg.n_queries);
        for i in 0..cfg.n_queries {
            // Queries drawn from same mixture but with independent noise —
            // realistic near-cluster queries, avoiding trivial exact hits.
            let cid = (i * 7 + 3) % cfg.k_clusters;
            queries.push(perturb(&centres[cid], cfg.noise_std * 1.5, &mut rng));
        }

        Self { db, queries, cfg }
    }

    pub fn ground_truth_mips(&self, k: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::with_capacity(self.queries.len());
        for q in &self.queries {
            let mut scored: Vec<(f32, u32)> = self
                .db
                .iter()
                .enumerate()
                .map(|(i, x)| (dot(q, x), i as u32))
                .collect();
            // Descending inner product — top-k MIPS.
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            out.push(scored.iter().take(k).map(|(_, i)| *i).collect());
        }
        out
    }
}

// ─── low-dependency RNG + helpers ─────────────────────────────────────────

pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut z = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.0 = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f32(&mut self) -> f32 {
        // Uniform in [0, 1).
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    pub fn next_gauss(&mut self) -> f32 {
        // Box-Muller.
        let u1 = self.next_f32().max(1e-9);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn rand_unit_vector(rng: &mut SplitMix64, dim: usize) -> Vector {
    let mut v = vec![0.0_f32; dim];
    for x in v.iter_mut() {
        *x = rng.next_gauss();
    }
    let n = norm(&v).max(1e-9);
    for x in v.iter_mut() {
        *x /= n;
    }
    v
}

fn perturb(centre: &[f32], std: f32, rng: &mut SplitMix64) -> Vector {
    centre
        .iter()
        .map(|&c| c + std * rng.next_gauss())
        .collect()
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}
