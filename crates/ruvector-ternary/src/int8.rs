//! Int8 scalar quantization baseline.
//!
//! Each coordinate is mapped affine into `i8` using a per-vector min/max.
//! Distance is the (unscaled) L2 on the i8 codes cast to i32 accumulator —
//! monotone with true L2 in the limit of no clipping, and the standard way
//! DiskANN/Weaviate/Milvus present "int8" for recall comparisons.

use crate::{Distance, Encoder};

#[derive(Clone, Debug)]
pub struct Int8Code {
    pub bytes: Vec<i8>,
    pub dim: usize,
}

impl Int8Code {
    pub fn bytes_len(&self) -> usize {
        self.bytes.len()
    }
}

/// Int8 scalar quantizer with a *global* symmetric scale so codes from
/// different vectors are directly comparable. `abs_max` is the fp32
/// magnitude that saturates to +/-127. For unit-variance Gaussians the
/// canonical setting is `abs_max = 4.0` (four sigma, <1e-4 clip rate).
#[derive(Clone, Copy, Debug)]
pub struct Int8Encoder {
    pub dim: usize,
    pub abs_max: f32,
}

impl Int8Encoder {
    pub fn new(dim: usize) -> Self {
        Self { dim, abs_max: 4.0 }
    }

    pub fn with_abs_max(dim: usize, abs_max: f32) -> Self {
        Self { dim, abs_max }
    }
}

impl Encoder for Int8Encoder {
    type Code = Int8Code;

    fn encode(&self, v: &[f32]) -> Self::Code {
        assert_eq!(v.len(), self.dim);
        let scale = 127.0 / self.abs_max.max(1e-12);
        let mut out = Vec::with_capacity(self.dim);
        for &x in v {
            let q = (x * scale).round().clamp(-127.0, 127.0);
            out.push(q as i8);
        }
        Int8Code { bytes: out, dim: self.dim }
    }

    fn bytes_per_code(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &'static str {
        "int8"
    }
}

#[derive(Default, Clone, Copy)]
pub struct Int8Distance;

impl Distance for Int8Distance {
    type Code = Int8Code;

    /// Unsigned L2^2 on i8 lanes. Fits in u32 for dim <= ~65k.
    #[inline]
    fn dist(&self, a: &Self::Code, b: &Self::Code) -> u32 {
        debug_assert_eq!(a.bytes.len(), b.bytes.len());
        let mut acc: i64 = 0;
        for i in 0..a.bytes.len() {
            let d = a.bytes[i] as i32 - b.bytes[i] as i32;
            acc += (d * d) as i64;
        }
        acc as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_distance_zero() {
        let e = Int8Encoder::new(8);
        let v = vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
        let c = e.encode(&v);
        let d = Int8Distance;
        assert_eq!(d.dist(&c, &c), 0);
    }

    #[test]
    fn distance_is_monotone_with_l2_on_random() {
        // For 10 random pairs, compare int8 ordering vs exact L2 ordering
        // relative to a fixed anchor.
        use crate::{l2, seeded_rng};
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = seeded_rng(11);
        let dim = 64;
        let e = Int8Encoder::new(dim);
        let d = Int8Distance;
        let anchor: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
        let ca = e.encode(&anchor);
        let mut pairs = vec![];
        for _ in 0..10 {
            let x: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(&mut rng)).collect();
            let cx = e.encode(&x);
            pairs.push((l2(&anchor, &x), d.dist(&ca, &cx)));
        }
        // Sort by exact, check int8 order preserves the top half.
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Rank correlation: at least 7/10 pairs adjacent in the same order.
        let mut agree = 0;
        for w in pairs.windows(2) {
            if w[0].1 <= w[1].1 {
                agree += 1;
            }
        }
        assert!(agree >= 7, "monotone in {}/9 adjacent pairs", agree);
    }
}
