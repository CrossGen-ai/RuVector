//! Deterministic PRNG (SplitMix64 + xoshiro256**) and synthetic
//! clustered dataset generator. Standalone so the benchmark reproduces
//! bit-for-bit without pulling `rand`.

pub struct Rng {
    s: [u64; 4],
}

fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut s = seed;
        let a = splitmix64(&mut s);
        let b = splitmix64(&mut s);
        let c = splitmix64(&mut s);
        let d = splitmix64(&mut s);
        Rng { s: [a, b, c, d] }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, bound).
    pub fn next_bounded(&mut self, bound: u32) -> u32 {
        (self.next_u64() as u32) % bound
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
    }

    /// Standard-normal via Box–Muller.
    pub fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-9);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        r * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Generate a clustered dataset: `n_clusters` centres placed on the
/// unit sphere, each spawning `n / n_clusters` samples with Gaussian
/// noise (std = `jitter`). This mimics real embeddings — not uniform
/// noise — so the coarse quantizer has structure to exploit.
pub fn make_clustered(n: usize, d: usize, n_clusters: usize, jitter: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut centres = vec![0.0f32; n_clusters * d];
    for c in 0..n_clusters {
        let mut norm = 0.0f32;
        for j in 0..d {
            let v = rng.next_gauss();
            centres[c * d + j] = v;
            norm += v * v;
        }
        let inv = 1.0 / norm.sqrt().max(1e-12);
        for j in 0..d {
            centres[c * d + j] *= inv;
        }
    }
    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let c = rng.next_bounded(n_clusters as u32) as usize;
        for j in 0..d {
            data[i * d + j] = centres[c * d + j] + jitter * rng.next_gauss();
        }
    }
    data
}

/// Draw queries from the same mixture but with a slightly larger
/// jitter — simulates a mild train/test distribution shift.
pub fn make_queries(n: usize, d: usize, n_clusters: usize, jitter: f32, seed: u64) -> Vec<f32> {
    make_clustered(n, d, n_clusters, jitter, seed)
}
