//! LVQ-4x8: two-stage residual quantization.
//!
//! Stage 1: LVQ-4 quantize the raw vector -> primary code + reconstruction.
//! Stage 2: LVQ-8 quantize the *residual* `r = v - decode(primary)`.
//!
//! Rationale (SVS paper, §4.2): the primary 4-bit code drives cheap
//! candidate generation across the whole corpus; the 8-bit residual gives
//! a nearly-fp32 refinement pass on a small candidate list.
//!
//! Storage per vector: `ceil(d/2) + d + 16` bytes (LVQ4 + LVQ8 + two
//! affine metadata pairs).
//!
//! Asymmetric distance here is the exact sum: since
//! `v_hat = decode(primary) + decode(residual)`, we accumulate against the
//! query as `sum_i (q_i - (primary_hat_i + residual_hat_i))^2`. That gives
//! the same recall as decoding-then-L2 while keeping the code path SIMD-
//! friendly (no allocations).

use crate::{Lvq4, Lvq4Code, Lvq8, Lvq8Code, Quantizer};

#[derive(Clone, Debug)]
pub struct Lvq4x8Code {
    pub primary: Lvq4Code,
    pub residual: Lvq8Code,
}

pub struct Lvq4x8 {
    dim: usize,
    lvq4: Lvq4,
    lvq8: Lvq8,
}

impl Lvq4x8 {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            lvq4: Lvq4::new(dim),
            lvq8: Lvq8::new(dim),
        }
    }
}

impl Quantizer for Lvq4x8 {
    type Code = Lvq4x8Code;

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, v: &[f32]) -> Self::Code {
        assert_eq!(v.len(), self.dim, "LVQ4x8.encode: dim mismatch");
        let primary = self.lvq4.encode(v);
        let primary_hat = self.lvq4.decode(&primary);
        let residual: Vec<f32> = v.iter().zip(primary_hat.iter()).map(|(a, b)| a - b).collect();
        let residual_code = self.lvq8.encode(&residual);
        Lvq4x8Code {
            primary,
            residual: residual_code,
        }
    }

    fn decode(&self, c: &Self::Code) -> Vec<f32> {
        let p = self.lvq4.decode(&c.primary);
        let r = self.lvq8.decode(&c.residual);
        p.iter().zip(r.iter()).map(|(a, b)| a + b).collect()
    }

    fn asymmetric_l2_sq(&self, q: &[f32], c: &Self::Code) -> f32 {
        debug_assert_eq!(q.len(), self.dim);
        let mut acc = 0.0f32;
        let plo = c.primary.lo;
        let pscale = c.primary.scale;
        let rlo = c.residual.lo;
        let rscale = c.residual.scale;
        for i in 0..self.dim {
            let pn = Lvq4::read_nibble(&c.primary.packed, i);
            let rn = c.residual.codes[i];
            let phat = plo + (pn as f32) * pscale;
            let rhat = rlo + (rn as f32) * rscale;
            let vhat = phat + rhat;
            let d = q[i] - vhat;
            acc += d * d;
        }
        acc
    }

    fn bytes_per_code(&self) -> usize {
        self.lvq4.bytes_per_code() + self.lvq8.bytes_per_code()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l2_sq_f32;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn residual_reduces_reconstruction_error() {
        // Residual layer should strictly improve reconstruction error vs
        // primary alone on random-ish data.
        let mut rng = StdRng::seed_from_u64(9);
        let d = 128;
        let q = Lvq4x8::new(d);
        let q4 = Lvq4::new(d);
        for _ in 0..8 {
            let v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let c4 = q4.encode(&v);
            let c48 = q.encode(&v);
            let err4 = l2_sq_f32(&v, &q4.decode(&c4));
            let err48 = l2_sq_f32(&v, &q.decode(&c48));
            assert!(err48 <= err4, "residual didn't help: {err48} vs {err4}");
        }
    }

    #[test]
    fn asymmetric_matches_decoded_l2_lvq4x8() {
        let mut rng = StdRng::seed_from_u64(15);
        let d = 96;
        let q = Lvq4x8::new(d);
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
}
