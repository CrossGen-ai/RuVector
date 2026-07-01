//! Variance-adaptive scalar quantizer.
//!
//! Each dimension `i` gets its own uniform quantizer over `[min_i, max_i]`
//! (calibrated from training data) with `bits[i]` bits of precision. Codes
//! for consecutive dims are bit-packed MSB-first into a single byte stream.

use crate::allocator::{allocate, expected_distortion};
use crate::{describe_dims, FasqError, Quantizer};

#[derive(Debug, Clone)]
pub struct Fasq {
    pub dim: usize,
    pub bits: Vec<u8>,       // per-dim bit widths
    pub mins: Vec<f32>,      // per-dim range floor
    pub scales: Vec<f32>,    // per-dim (max - min) / (2^bits - 1); zero if flat
    pub bit_offsets: Vec<usize>, // starting bit offset per dim within packed record
    pub total_bits: usize,
    pub expected_distortion: f64,
}

impl Fasq {
    /// Train a FASQ quantizer from a slice of training vectors.
    ///
    /// * `b_avg` — target average bits per dimension.
    /// * `b_lo`, `b_hi` — per-dim bit bounds.
    pub fn train(
        train: &[Vec<f32>],
        b_avg: f32,
        b_lo: u8,
        b_hi: u8,
    ) -> Result<Self, FasqError> {
        let stats = describe_dims(train)?;
        let dim = stats.len();
        let sigma2: Vec<f32> = stats.iter().map(|s| s.variance).collect();
        let bits = allocate(&sigma2, b_avg, b_lo, b_hi)?;

        let mut mins = vec![0.0f32; dim];
        let mut scales = vec![0.0f32; dim];
        for i in 0..dim {
            mins[i] = stats[i].min;
            let range = stats[i].max - stats[i].min;
            let levels = (1u64 << bits[i]) - 1;
            scales[i] = if range > 0.0 && levels > 0 {
                range / levels as f32
            } else {
                0.0
            };
        }

        let mut bit_offsets = vec![0usize; dim];
        let mut off = 0usize;
        for i in 0..dim {
            bit_offsets[i] = off;
            off += bits[i] as usize;
        }
        let total_bits = off;
        let expected_distortion = expected_distortion(&sigma2, &bits);
        Ok(Self { dim, bits, mins, scales, bit_offsets, total_bits, expected_distortion })
    }

    #[inline]
    fn quantize_dim(&self, i: usize, x: f32) -> u32 {
        if self.scales[i] == 0.0 { return 0; }
        let levels = (1u64 << self.bits[i]) - 1;
        let mut q = ((x - self.mins[i]) / self.scales[i]).round();
        if q < 0.0 { q = 0.0; }
        if q > levels as f32 { q = levels as f32; }
        q as u32
    }

    #[inline]
    fn dequantize_dim(&self, i: usize, q: u32) -> f32 {
        self.mins[i] + (q as f32) * self.scales[i]
    }
}

impl Quantizer for Fasq {
    fn bits_per_vector(&self) -> usize { self.total_bits }

    fn encode(&self, v: &[f32], out: &mut Vec<u8>) -> Result<usize, FasqError> {
        if v.len() != self.dim {
            return Err(FasqError::DimMismatch { expected: self.dim, got: v.len() });
        }
        let nbytes = self.bytes_per_vector();
        let start = out.len();
        out.resize(start + nbytes, 0);
        let buf = &mut out[start..];
        for i in 0..self.dim {
            let code = self.quantize_dim(i, v[i]);
            write_bits(buf, self.bit_offsets[i], self.bits[i], code);
        }
        Ok(nbytes)
    }

    fn decode(&self, bytes: &[u8], out: &mut [f32]) -> Result<(), FasqError> {
        if out.len() != self.dim {
            return Err(FasqError::DimMismatch { expected: self.dim, got: out.len() });
        }
        if bytes.len() < self.bytes_per_vector() {
            return Err(FasqError::DimMismatch {
                expected: self.bytes_per_vector(),
                got: bytes.len(),
            });
        }
        for i in 0..self.dim {
            let code = read_bits(bytes, self.bit_offsets[i], self.bits[i]);
            out[i] = self.dequantize_dim(i, code);
        }
        Ok(())
    }
}

/// Bit-width-safe mask: returns `(1u64 << n) - 1` for `n <= 64`.
///
/// Rust's `<<` masks the shift amount modulo the type width in release, so
/// `1u8 << 8` becomes `1u8 << 0 == 1` rather than `0`. We route through u64
/// and special-case `n == 64` to keep behaviour independent of build profile.
#[inline]
fn low_mask_u32(n: usize) -> u32 {
    debug_assert!(n <= 32);
    if n == 32 { u32::MAX } else { (1u32 << n) - 1 }
}

/// Write `nbits` (≤32) of `value` into `buf` starting at bit `offset`
/// (MSB-first ordering, byte 0 holds the most-significant bits).
fn write_bits(buf: &mut [u8], offset: usize, nbits: u8, value: u32) {
    let mut remaining = nbits as usize;
    let mut v = value & low_mask_u32(nbits as usize);
    let mut bit_pos = offset;
    while remaining > 0 {
        let byte_idx = bit_pos / 8;
        let bit_in_byte = bit_pos % 8;
        let free_in_byte = 8 - bit_in_byte;
        let take = remaining.min(free_in_byte);
        let shift_down = remaining - take;
        let chunk_u32 = (v >> shift_down) & low_mask_u32(take);
        let byte_shift = free_in_byte - take;
        buf[byte_idx] |= (chunk_u32 as u8) << byte_shift;
        // Clear bits we just wrote so they aren't re-written next iteration.
        v &= low_mask_u32(shift_down);
        bit_pos += take;
        remaining -= take;
    }
}

/// Read `nbits` (≤32) starting at bit `offset` (MSB-first ordering).
fn read_bits(buf: &[u8], offset: usize, nbits: u8) -> u32 {
    let mut remaining = nbits as usize;
    let mut bit_pos = offset;
    let mut out: u32 = 0;
    while remaining > 0 {
        let byte_idx = bit_pos / 8;
        let bit_in_byte = bit_pos % 8;
        let free_in_byte = 8 - bit_in_byte;
        let take = remaining.min(free_in_byte);
        let byte_shift = free_in_byte - take;
        let chunk = ((buf[byte_idx] >> byte_shift) as u32) & low_mask_u32(take);
        out = (out << take) | chunk;
        bit_pos += take;
        remaining -= take;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_io_roundtrip() {
        let mut buf = vec![0u8; 16];
        write_bits(&mut buf, 0, 3, 0b101);
        write_bits(&mut buf, 3, 5, 0b11010);
        write_bits(&mut buf, 8, 12, 0b1010_1100_1111);
        assert_eq!(read_bits(&buf, 0, 3), 0b101);
        assert_eq!(read_bits(&buf, 3, 5), 0b11010);
        assert_eq!(read_bits(&buf, 8, 12), 0b1010_1100_1111);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let train: Vec<Vec<f32>> = (0..500)
            .map(|i| (0..16).map(|j| ((i + j) % 17) as f32).collect())
            .collect();
        let q = Fasq::train(&train, 6.0, 2, 8).unwrap();
        let mut code = Vec::new();
        let mut recon = vec![0.0; 16];
        for v in &train {
            code.clear();
            q.encode(v, &mut code).unwrap();
            q.decode(&code, &mut recon).unwrap();
            for i in 0..16 {
                // Reconstruction should be within one quantization step.
                let step = q.scales[i];
                assert!((v[i] - recon[i]).abs() <= step + 1e-4,
                    "dim {i} exceeds step {step}: v={} recon={}", v[i], recon[i]);
            }
        }
    }
}
