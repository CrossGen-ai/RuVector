//! Squared L2 distance — kept inline-friendly with no SIMD intrinsics so the
//! benchmark numbers reflect honest scalar Rust performance.

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Deterministic LCG that lets the benchmark generate the same dataset on every
/// run without pulling in `rand`.
pub struct Lcg(pub u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32)
    }
    /// Box–Muller transform — adequate Gaussian sampling for benchmarks.
    pub fn next_gauss(&mut self) -> f32 {
        let u1 = (self.next_f32().max(1e-9)).min(1.0 - 1e-9);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}
