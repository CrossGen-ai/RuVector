//! B-bit uniform symmetric quantization of rotated unit vectors.
//!
//! We rotate + unit-normalise each database vector, then encode every
//! coordinate with `B ∈ {1, 2, 4, 8}` bits over a symmetric grid centred
//! at 0. The reconstruction levels are stored once in the quantizer.
//!
//! Layout: codes are packed little-endian into a `Vec<u8>` — bit `b` of
//! dimension `d` lives at global bit `d * B + b`. This keeps decode simple
//! and portable at the cost of a small shift/mask per dim during scan.

use serde::{Deserialize, Serialize};

use crate::error::ExtRabitqError;

/// Multi-bit code for a single rotated unit vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedCode {
    /// Packed little-endian code bits.
    pub bytes: Vec<u8>,
    /// Original vector norm before unit-normalisation (needed for L2 decode).
    pub norm: f32,
}

/// Reusable quantizer: owns the reconstruction level table and per-dim
/// packing routine.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedQuantizer {
    dim: usize,
    bits: u32,
    /// `2^bits` reconstruction values in `[-1, 1]`. `levels[k]` is the value
    /// used both at encode-time (nearest-neighbour on the grid) and at
    /// scan-time (LUT lookup).
    levels: Vec<f32>,
}

impl ExtendedQuantizer {
    /// Build a `bits`-per-dim symmetric uniform quantizer.
    pub fn new(dim: usize, bits: u32) -> Result<Self, ExtRabitqError> {
        if dim == 0 {
            return Err(ExtRabitqError::InvalidDim { dim });
        }
        if !matches!(bits, 1 | 2 | 4 | 8) {
            return Err(ExtRabitqError::UnsupportedBits { bits });
        }
        let n = 1u32 << bits;
        // Symmetric uniform grid: levels[k] = 2 * (k + 0.5) / n - 1
        // for k = 0..n. Range [-1 + 1/n, 1 - 1/n]. Zero-mean, symmetric.
        let levels: Vec<f32> = (0..n)
            .map(|k| (2.0 * (k as f32 + 0.5) / n as f32) - 1.0)
            .collect();
        Ok(Self { dim, bits, levels })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn bits(&self) -> u32 {
        self.bits
    }
    pub fn levels(&self) -> &[f32] {
        &self.levels
    }

    /// Estimated bytes per code (excluding the `norm` field).
    pub fn code_bytes(&self) -> usize {
        (self.dim * self.bits as usize + 7) / 8
    }

    /// Encode a length-`dim` vector.
    ///
    /// Steps: (a) capture original norm, (b) unit-normalise, (c) for each
    /// dimension pick the nearest reconstruction level, (d) pack.
    pub fn encode(&self, rotated: &[f32]) -> Result<ExtendedCode, ExtRabitqError> {
        if rotated.len() != self.dim {
            return Err(ExtRabitqError::DimMismatch {
                expected: self.dim,
                actual: rotated.len(),
            });
        }
        let norm_sq: f32 = rotated.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };

        let n_levels = 1usize << self.bits;
        let mut bytes = vec![0u8; self.code_bytes()];
        for (d, &v) in rotated.iter().enumerate() {
            let u = (v * inv).clamp(-1.0, 1.0);
            // Uniform grid: nearest level index is floor((u + 1) * n / 2).
            let idx_f = (u + 1.0) * (n_levels as f32) * 0.5;
            let mut idx = idx_f as isize;
            if idx < 0 {
                idx = 0;
            }
            if idx >= n_levels as isize {
                idx = n_levels as isize - 1;
            }
            let code = idx as u32;
            let bit_off = d * self.bits as usize;
            let byte_off = bit_off / 8;
            let shift = (bit_off % 8) as u32;
            // Handle B ∈ {1,2,4,8} — max shift+bits ≤ 8 for B∈{1,2,4} in a
            // byte-aligned dim, and B=8 always aligns. For B=2 and B=4 we
            // are guaranteed byte-aligned because 8 % B == 0. For B=1 same.
            debug_assert!(shift + self.bits <= 8, "packing overflow");
            let mask = ((1u32 << self.bits) - 1) as u8;
            bytes[byte_off] |= ((code as u8) & mask) << shift;
        }
        Ok(ExtendedCode { bytes, norm })
    }

    /// Decode a code back to unit-norm approximate rotated vector.
    /// Useful for tests + reranking; the hot path uses `scan` instead.
    pub fn decode(&self, code: &ExtendedCode) -> Vec<f32> {
        let n_levels = 1usize << self.bits;
        let mask = ((1u32 << self.bits) - 1) as u8;
        let mut out = vec![0f32; self.dim];
        for d in 0..self.dim {
            let bit_off = d * self.bits as usize;
            let byte_off = bit_off / 8;
            let shift = (bit_off % 8) as u32;
            let idx = ((code.bytes[byte_off] >> shift) & mask) as usize;
            debug_assert!(idx < n_levels);
            out[d] = self.levels[idx];
        }
        out
    }

    /// Extract per-dim level indices (for LUT-based scan).
    pub fn indices(&self, code: &ExtendedCode) -> Vec<u8> {
        let mask = ((1u32 << self.bits) - 1) as u8;
        let mut out = vec![0u8; self.dim];
        for d in 0..self.dim {
            let bit_off = d * self.bits as usize;
            let byte_off = bit_off / 8;
            let shift = (bit_off % 8) as u32;
            out[d] = (code.bytes[byte_off] >> shift) & mask;
        }
        out
    }
}

/// Pack per-dim reconstruction values for a batch: convenient for the LUT
/// scan when we prefer a densely stored `Vec<u8>` of indices instead of
/// packed bytes.
pub fn indices_from_codes(q: &ExtendedQuantizer, codes: &[ExtendedCode]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(codes.len() * q.dim());
    for c in codes {
        buf.extend_from_slice(&q.indices(c));
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_bits_2() {
        let q = ExtendedQuantizer::new(4, 2).unwrap();
        let v = vec![0.1, -0.5, 0.7, -0.9];
        let code = q.encode(&v).unwrap();
        let dec = q.decode(&code);
        // Reconstruction should be within max grid spacing (2/n_levels).
        let max_err = 2.0 / 4.0; // n_levels = 4
        // Compare after re-normalising the input.
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for (a, b) in dec.iter().zip(v.iter()) {
            assert!((*a - b / n).abs() <= max_err + 1e-3);
        }
    }

    #[test]
    fn code_size_bits_4() {
        let q = ExtendedQuantizer::new(128, 4).unwrap();
        assert_eq!(q.code_bytes(), 64);
    }
}
