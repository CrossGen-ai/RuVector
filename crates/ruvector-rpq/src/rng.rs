//! Deterministic xorshift64* PRNG — no external crates.

/// Tiny xorshift64* PRNG. Deterministic; suitable for benchmarks and tests.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid degenerate zero state.
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }
    /// Approximate standard normal via Box-Muller. Cheap and adequate here.
    pub fn next_gauss(&mut self) -> f32 {
        let mut u1 = self.next_f32();
        if u1 < 1e-7 {
            u1 = 1e-7;
        }
        let u2 = self.next_f32();
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0 * core::f32::consts::PI * u2;
        r * theta.cos()
    }
}
