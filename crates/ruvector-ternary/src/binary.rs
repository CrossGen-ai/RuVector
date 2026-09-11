//! 1-bit binary (sign) encoding — the classical baseline.
//!
//! Each coordinate is encoded as the bit `sign(x) >= 0`. Distance is the
//! Hamming distance between two bit-packed codes: `popcount(a ^ b)`.

use crate::{words_for, Distance, Encoder};

/// Bit-packed 1-bit code.
#[derive(Clone, Debug)]
pub struct BinaryCode {
    /// Words holding the sign bits (bit i of word `i/64`).
    pub sign: Vec<u64>,
    /// Number of dimensions actually populated (informational).
    pub dim: usize,
}

impl BinaryCode {
    pub fn bytes(&self) -> usize {
        self.sign.len() * 8
    }
}

/// Sign-only encoder. Coordinates equal to zero encode as `+1` bit; this
/// matches the standard "sign(x) >= 0" convention and keeps the baseline
/// deterministic.
#[derive(Default, Clone, Copy)]
pub struct BinaryEncoder {
    pub dim: usize,
}

impl BinaryEncoder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Encoder for BinaryEncoder {
    type Code = BinaryCode;

    fn encode(&self, v: &[f32]) -> Self::Code {
        assert_eq!(v.len(), self.dim, "dim mismatch");
        let mut sign = vec![0u64; words_for(self.dim)];
        for (i, &x) in v.iter().enumerate() {
            if x >= 0.0 {
                sign[i / 64] |= 1u64 << (i % 64);
            }
        }
        BinaryCode { sign, dim: self.dim }
    }

    fn bytes_per_code(&self) -> usize {
        words_for(self.dim) * 8
    }

    fn name(&self) -> &'static str {
        "binary"
    }
}

/// Hamming distance on bit-packed sign codes.
#[derive(Default, Clone, Copy)]
pub struct BinaryDistance;

impl Distance for BinaryDistance {
    type Code = BinaryCode;

    #[inline]
    fn dist(&self, a: &Self::Code, b: &Self::Code) -> u32 {
        debug_assert_eq!(a.sign.len(), b.sign.len());
        let mut acc: u32 = 0;
        for i in 0..a.sign.len() {
            acc += (a.sign[i] ^ b.sign[i]).count_ones();
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_signs() {
        let e = BinaryEncoder::new(4);
        let v = vec![1.0, -1.0, 0.5, -0.5];
        let c = e.encode(&v);
        // bits: 1 0 1 0 => 0b0101 in LSB-first packing => value 5
        assert_eq!(c.sign[0] & 0xF, 0b0101);
    }

    #[test]
    fn hamming_symmetry_and_zero() {
        let e = BinaryEncoder::new(8);
        let d = BinaryDistance;
        let a = e.encode(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
        let b = e.encode(&[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
        assert_eq!(d.dist(&a, &a), 0);
        assert_eq!(d.dist(&a, &b), 8);
        assert_eq!(d.dist(&a, &b), d.dist(&b, &a));
    }
}
