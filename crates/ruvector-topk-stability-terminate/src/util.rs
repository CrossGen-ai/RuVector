//! Deterministic RNG + vector helpers. Keeps the PoC dependency-free so
//! `cargo build -p ruvector-topk-stability-terminate` needs no network.

/// Splitmix64 — small, fast, deterministic 64-bit PRNG. Public-domain
/// algorithm (Vigna, 2015). Adequate for research-grade synthetic vectors;
/// not a cryptographic RNG.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        // Avoid all-zero state.
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// Uniform f32 in [-1, 1).
    pub fn next_f32(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32; // 24 bits
        (u as f32) / ((1u32 << 23) as f32) - 1.0
    }
    pub fn next_range(&mut self, n: u32) -> u32 {
        (self.next_u64() as u32) % n.max(1)
    }
}

/// Generate `n` L2-normalized f32 vectors of `dim` dimensions, deterministic
/// under `seed`. Vectors are laid out row-major in the returned Vec.
pub fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut v = vec![0f32; n * dim];
    for i in 0..n {
        let s = &mut v[i * dim..(i + 1) * dim];
        let mut norm = 0f32;
        for x in s.iter_mut() {
            *x = rng.next_f32();
            norm += *x * *x;
        }
        let inv = 1.0 / norm.sqrt().max(1e-12);
        for x in s.iter_mut() {
            *x *= inv;
        }
    }
    v
}

/// Squared Euclidean distance between two `dim`-length slices. Panics if
/// lengths differ — this is intentionally a tight kernel called in a hot loop.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn random_unit_vectors_are_unit_norm() {
        let dim = 32;
        let v = random_unit_vectors(10, dim, 7);
        for i in 0..10 {
            let s = &v[i * dim..(i + 1) * dim];
            let n2: f32 = s.iter().map(|x| x * x).sum();
            assert!((n2 - 1.0).abs() < 1e-4, "row {i} not unit: {n2}");
        }
    }

    #[test]
    fn sq_l2_zero_on_self() {
        let v = random_unit_vectors(1, 16, 3);
        assert!(sq_l2(&v, &v) < 1e-8);
    }
}
