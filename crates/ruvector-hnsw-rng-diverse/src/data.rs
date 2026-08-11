//! Deterministic synthetic-data generators for the benchmark. Pure std RNG
//! (xorshift64*) — reproducible across runs without a rand dep.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_mul(0x2545_F491_4F6C_DD1D) | 1 }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// U[0,1).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Box–Muller standard normal.
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.f32() as f64).max(1e-9);
        let u2 = self.f32() as f64;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z as f32
    }
}

/// Gaussian-mixture dataset: `c` isotropic clusters in `R^d`.
pub fn mixture(n: usize, d: usize, c: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    // Cluster centers on a hypercube scaled to keep them separated.
    // Centers moderately separated so cross-cluster candidates still appear
    // within reasonable ef_build; within-cluster spread is large enough that
    // some cross-cluster edges are competitive under RNG pruning.
    let centers: Vec<Vec<f32>> = (0..c)
        .map(|_| (0..d).map(|_| rng.normal() * 1.2).collect())
        .collect();
    (0..n)
        .map(|i| {
            let ci = i % c;
            (0..d).map(|j| centers[ci][j] + rng.normal() * 0.9).collect()
        })
        .collect()
}
