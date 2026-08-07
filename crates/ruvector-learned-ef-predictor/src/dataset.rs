//! Deterministic synthetic clustered dataset (Gaussian mixtures) for HNSW
//! benchmarking. No external deps: a tiny SplitMix64 PRNG feeds a
//! Box-Muller Gaussian.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    pub fn uniform01(&mut self) -> f32 {
        // 24-bit mantissa unit uniform.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
    #[inline]
    pub fn gauss(&mut self) -> f32 {
        // Box-Muller. Two uniforms → one normal (discard the other).
        let mut u1 = self.uniform01();
        if u1 < 1e-7 { u1 = 1e-7; }
        let u2 = self.uniform01();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    pub fn choice(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

/// Build a clustered dataset: `n_clusters` centers on the unit hypercube,
/// each vector = center + `sigma` * N(0, I).
pub fn make_clusters(
    n_vectors: usize,
    dim: usize,
    n_clusters: usize,
    sigma: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);

    // Generate cluster centers with wider spread.
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gauss() * 3.0).collect())
        .collect();

    let mut out = Vec::with_capacity(n_vectors);
    for _ in 0..n_vectors {
        let c = &centers[rng.choice(n_clusters)];
        let v: Vec<f32> = c.iter().map(|x| x + rng.gauss() * sigma).collect();
        out.push(v);
    }
    out
}

/// Build a query set drawn from the same generator (guarantees realistic
/// in-distribution queries).
pub fn make_queries(n: usize, dim: usize, n_clusters: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    make_clusters(n, dim, n_clusters, sigma, seed)
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
