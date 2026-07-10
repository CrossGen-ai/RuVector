//! Deterministic xorshift64 RNG for reproducible builds.
//!
//! Not cryptographic — we only need reproducibility for benchmarks and tests.

/// 64-bit xorshift generator (Marsaglia 2003). Deterministic, seedable, fast.
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Create a new generator. Seed is coerced to non-zero to avoid the
    /// all-zero fixed point of xorshift.
    pub fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state }
    }

    /// Next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Use top 24 bits for float mantissa
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform f32 in [-1, 1).
    pub fn next_signed_f32(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }

    /// Uniform usize in [0, upper).
    pub fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper.max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_from_seed() {
        let mut a = Xorshift64::new(42);
        let mut b = Xorshift64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn zero_seed_is_reseeded() {
        let mut r = Xorshift64::new(0);
        // Not stuck at zero.
        assert_ne!(r.next_u64(), 0);
    }

    #[test]
    fn f32_range() {
        let mut r = Xorshift64::new(7);
        for _ in 0..1000 {
            let x = r.next_f32();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
