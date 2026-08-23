//! Tiny deterministic PRNG (xorshift64*) and Gaussian sampler (Box-Muller).
//! Zero-dep so this research crate never fights workspace resolver drift.

/// Xorshift64* PRNG.
pub struct Xor64 { state: u64 }

impl Xor64 {
    /// New PRNG from seed (seed must be non-zero; 1 is used if zero passed).
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }
    /// Next raw u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1Du64)
    }
    /// Uniform f32 in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        // Take top 24 bits.
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }
    /// Standard-normal f32 via Box-Muller. Wastes one draw per call for simplicity.
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos()
    }
}
