//! LVQ-8: 8-bit locally-adaptive scalar quantization.
//!
//! For each vector `v` (dim = `d`):
//! - Compute `lo = min(v)`, `hi = max(v)`.
//! - `scale = (hi - lo) / 255`. Guard `scale <= 0` -> use `scale = 1`.
//! - Each component `v[i]` is stored as
//!   `q[i] = round((v[i] - lo) / scale)` clamped to `[0, 255]`.
//! - Reconstruction: `v_hat[i] = lo + q[i] * scale`.
//!
//! Storage per vector: `d` bytes for the code, plus 8 bytes for `(lo, scale)`.
//! Overall ratio vs fp32: `(d + 8) / (4d)` — approaches 0.25 (4x compression)
//! for typical `d >= 64`.

use crate::Quantizer;

#[derive(Clone, Debug)]
pub struct Lvq8Code {
    pub lo: f32,
    pub scale: f32,
    pub codes: Vec<u8>, // length = dim
}

pub struct Lvq8 {
    dim: usize,
}

impl Lvq8 {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "LVQ8 dim must be > 0");
        Self { dim }
    }
}

impl Quantizer for Lvq8 {
    type Code = Lvq8Code;

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, v: &[f32]) -> Self::Code {
        assert_eq!(v.len(), self.dim, "LVQ8.encode: dim mismatch");
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &x in v {
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
        }
        let span = hi - lo;
        let scale = if span > 0.0 { span / 255.0 } else { 1.0 };
        let inv = 1.0 / scale;
        let mut codes = Vec::with_capacity(self.dim);
        for &x in v {
            let q = ((x - lo) * inv).round().clamp(0.0, 255.0) as u8;
            codes.push(q);
        }
        Lvq8Code { lo, scale, codes }
    }

    fn decode(&self, c: &Self::Code) -> Vec<f32> {
        c.codes.iter().map(|&q| c.lo + (q as f32) * c.scale).collect()
    }

    fn asymmetric_l2_sq(&self, q: &[f32], c: &Self::Code) -> f32 {
        debug_assert_eq!(q.len(), self.dim);
        debug_assert_eq!(c.codes.len(), self.dim);
        let mut acc = 0.0f32;
        let lo = c.lo;
        let scale = c.scale;
        for i in 0..self.dim {
            let bhat = lo + (c.codes[i] as f32) * scale;
            let d = q[i] - bhat;
            acc += d * d;
        }
        acc
    }

    fn bytes_per_code(&self) -> usize {
        self.dim + 8 // codes + (lo, scale) fp32 pair
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l2_sq_f32;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn roundtrip_preserves_dim() {
        let q = Lvq8::new(64);
        let v: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let c = q.encode(&v);
        let vhat = q.decode(&c);
        assert_eq!(vhat.len(), 64);
    }

    #[test]
    fn constant_vector_round_trips_exactly() {
        let q = Lvq8::new(16);
        let v = vec![0.42f32; 16];
        let c = q.encode(&v);
        let vhat = q.decode(&c);
        for &x in &vhat {
            assert!((x - 0.42).abs() < 1e-5, "got {x}");
        }
    }

    #[test]
    fn asymmetric_matches_decoded_l2() {
        // Asymmetric distance must equal L2(q, decode(c)) exactly.
        let mut rng = StdRng::seed_from_u64(7);
        let d = 128;
        let q = Lvq8::new(d);
        for _ in 0..8 {
            let v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let qv: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let c = q.encode(&v);
            let via_decode = l2_sq_f32(&qv, &q.decode(&c));
            let asym = q.asymmetric_l2_sq(&qv, &c);
            assert!(
                (via_decode - asym).abs() / (via_decode + 1e-6) < 1e-5,
                "asym {asym} vs decoded {via_decode}",
            );
        }
    }

    #[test]
    fn quantization_error_bounded_by_scale() {
        // Per-component reconstruction error is bounded by scale/2.
        let mut rng = StdRng::seed_from_u64(3);
        let q = Lvq8::new(256);
        let v: Vec<f32> = (0..256).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let c = q.encode(&v);
        let vhat = q.decode(&c);
        for i in 0..256 {
            assert!(
                (v[i] - vhat[i]).abs() <= c.scale * 0.5 + 1e-5,
                "err {} > bound {}", (v[i] - vhat[i]).abs(), c.scale * 0.5
            );
        }
    }
}
