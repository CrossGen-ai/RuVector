//! LVQ-4: 4-bit locally-adaptive scalar quantization, two components per byte.
//!
//! Same per-vector affine encoding as LVQ8, but with 16 levels instead of 256.
//! Two 4-bit codes are packed into each output byte (low nibble = even index,
//! high nibble = odd index).
//!
//! Storage per vector: `ceil(d/2)` bytes for the code, plus 8 bytes for
//! `(lo, scale)`. For `d >= 64`, approaches 0.125 (8x compression from fp32).

use crate::Quantizer;

#[derive(Clone, Debug)]
pub struct Lvq4Code {
    pub lo: f32,
    pub scale: f32,
    pub packed: Vec<u8>, // length = ceil(dim / 2)
    pub dim: usize,      // logical dimension (needed to distinguish odd d)
}

pub struct Lvq4 {
    dim: usize,
}

impl Lvq4 {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "LVQ4 dim must be > 0");
        Self { dim }
    }

    #[inline]
    pub fn packed_len(dim: usize) -> usize {
        (dim + 1) / 2
    }

    /// Read the 4-bit code at logical index `i` from a packed byte slice.
    #[inline]
    pub fn read_nibble(packed: &[u8], i: usize) -> u8 {
        let byte = packed[i >> 1];
        if i & 1 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        }
    }
}

impl Quantizer for Lvq4 {
    type Code = Lvq4Code;

    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, v: &[f32]) -> Self::Code {
        assert_eq!(v.len(), self.dim, "LVQ4.encode: dim mismatch");
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
        let scale = if span > 0.0 { span / 15.0 } else { 1.0 };
        let inv = 1.0 / scale;

        let plen = Self::packed_len(self.dim);
        let mut packed = vec![0u8; plen];
        for i in 0..self.dim {
            let q = ((v[i] - lo) * inv).round().clamp(0.0, 15.0) as u8;
            if i & 1 == 0 {
                packed[i >> 1] |= q & 0x0F;
            } else {
                packed[i >> 1] |= (q & 0x0F) << 4;
            }
        }
        Lvq4Code {
            lo,
            scale,
            packed,
            dim: self.dim,
        }
    }

    fn decode(&self, c: &Self::Code) -> Vec<f32> {
        let mut out = Vec::with_capacity(c.dim);
        for i in 0..c.dim {
            let q = Self::read_nibble(&c.packed, i);
            out.push(c.lo + (q as f32) * c.scale);
        }
        out
    }

    fn asymmetric_l2_sq(&self, q: &[f32], c: &Self::Code) -> f32 {
        debug_assert_eq!(q.len(), self.dim);
        debug_assert_eq!(c.dim, self.dim);
        let mut acc = 0.0f32;
        let lo = c.lo;
        let scale = c.scale;
        for i in 0..self.dim {
            let nib = Self::read_nibble(&c.packed, i);
            let bhat = lo + (nib as f32) * scale;
            let d = q[i] - bhat;
            acc += d * d;
        }
        acc
    }

    fn bytes_per_code(&self) -> usize {
        Self::packed_len(self.dim) + 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::l2_sq_f32;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn packed_len_even_and_odd() {
        assert_eq!(Lvq4::packed_len(64), 32);
        assert_eq!(Lvq4::packed_len(65), 33);
        assert_eq!(Lvq4::packed_len(1), 1);
    }

    #[test]
    fn odd_dim_roundtrip_length() {
        let q = Lvq4::new(65);
        let v: Vec<f32> = (0..65).map(|i| i as f32 / 65.0).collect();
        let c = q.encode(&v);
        let vhat = q.decode(&c);
        assert_eq!(vhat.len(), 65);
    }

    #[test]
    fn asymmetric_matches_decoded_l2_lvq4() {
        let mut rng = StdRng::seed_from_u64(11);
        let d = 128;
        let q = Lvq4::new(d);
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
    fn error_bounded_by_scale_lvq4() {
        let mut rng = StdRng::seed_from_u64(4);
        let q = Lvq4::new(128);
        let v: Vec<f32> = (0..128).map(|_| rng.gen_range(-5.0..5.0)).collect();
        let c = q.encode(&v);
        let vhat = q.decode(&c);
        for i in 0..128 {
            assert!(
                (v[i] - vhat[i]).abs() <= c.scale * 0.5 + 1e-5,
                "err {} > bound {}", (v[i] - vhat[i]).abs(), c.scale * 0.5
            );
        }
    }
}
