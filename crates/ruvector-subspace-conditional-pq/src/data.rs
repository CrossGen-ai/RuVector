//! Deterministic synthetic data generators used by tests and benches.
//!
//! `gen_correlated` produces N D-dim vectors with controllable inter-subspace
//! correlation. We draw a low-rank latent z ~ N(0, I_r), project to R^D with
//! a fixed random mixing matrix A (so every output dimension is a linear
//! combination of the same r latents), then add ε·N(0, I_d) isotropic noise.
//! Smaller ε → more correlation across subspaces; this is exactly the regime
//! where SC-PQ should beat vanilla PQ.
//!
//! `gen_independent` draws N(0, I_d) vectors — subspaces are independent, so
//! SC-PQ should NOT improve over PQ (and we test for that in the doc's
//! "negative result" discussion).

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Standard normal via Box-Muller (uses two uniforms per call).
    pub fn randn(&mut self) -> f32 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-12);
        let r = (-2.0 * u1.ln()).sqrt();
        (r * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

pub fn gen_correlated(n: usize, d: usize, latent: usize, noise: f32, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    // Fixed mixing matrix A: d x latent.
    let mut a = vec![0f32; d * latent];
    for v in a.iter_mut() {
        *v = rng.randn();
    }
    let mut out = vec![0f32; n * d];
    let mut z = vec![0f32; latent];
    for i in 0..n {
        for v in z.iter_mut() {
            *v = rng.randn();
        }
        for j in 0..d {
            let mut s = 0f32;
            for r in 0..latent {
                s += a[j * latent + r] * z[r];
            }
            out[i * d + j] = s + noise * rng.randn();
        }
    }
    out
}

pub fn gen_independent(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut out = vec![0f32; n * d];
    for v in out.iter_mut() {
        *v = rng.randn();
    }
    out
}

pub fn mse(data: &[f32], recon: &[f32]) -> f64 {
    assert_eq!(data.len(), recon.len());
    let mut s = 0f64;
    for i in 0..data.len() {
        let d = (data[i] - recon[i]) as f64;
        s += d * d;
    }
    s / data.len() as f64
}
