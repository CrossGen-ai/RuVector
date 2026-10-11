//! Encoder: rotate -> normalise -> per-coordinate B-bit quantize.
//! Stores codes densely packed and keeps norms + reconstruction constants.

use crate::codebook::Codebook;
use crate::rotation::{rotate_to_padded, FastRotation};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedRaBitQ {
    pub bits: u8,
    pub dim: usize,
    pub padded_dim: usize,
    pub rotation: FastRotation,
    pub codebook: Codebook,
}

impl ExtendedRaBitQ {
    pub fn new(dim: usize, bits: u8, seed: u64) -> Result<Self> {
        if !(1..=8).contains(&bits) {
            return Err(Error::UnsupportedBits(bits));
        }
        let rotation = FastRotation::new(dim, seed);
        let padded_dim = rotation.padded_dim();
        let codebook = Codebook::gaussian(bits);
        Ok(Self {
            bits,
            dim,
            padded_dim,
            rotation,
            codebook,
        })
    }

    /// Encode a single vector. Returns (codes, l2_norm). Codes are packed
    /// little-endian: coordinate i occupies bits [i*B..i*B+B] of the stream.
    pub fn encode(&self, x: &[f32]) -> Result<(Vec<u8>, f32)> {
        if x.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: x.len(),
            });
        }
        let rotated = rotate_to_padded(&self.rotation, x);
        let norm = l2_norm(&rotated);
        let inv = if norm > 1e-20 { 1.0 / norm } else { 0.0 };
        // Convert each coordinate from N(0, 1/D) to N(0, 1) by multiplying by sqrt(D_padded)
        // (unit vectors in D-dim have per-coord variance 1/D, so sqrt(D) rescales to std-normal).
        let scale = (self.padded_dim as f32).sqrt();
        let mut codes_u8: Vec<u8> = Vec::with_capacity(self.padded_dim);
        for v in &rotated {
            let z = v * inv * scale;
            codes_u8.push(self.codebook.encode_scalar(z));
        }
        Ok((pack_bits(&codes_u8, self.bits), norm))
    }

    /// Decode codes into an approximate rotated **unit** vector (padded_dim).
    pub fn decode_unit(&self, codes: &[u8]) -> Vec<f32> {
        let scale = 1.0 / (self.padded_dim as f32).sqrt();
        unpack_bits(codes, self.bits, self.padded_dim)
            .into_iter()
            .map(|c| self.codebook.decode(c) * scale)
            .collect()
    }

    /// Prepare a query: rotate (no quantization). Length = padded_dim.
    pub fn prepare_query(&self, q: &[f32]) -> Result<Vec<f32>> {
        if q.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: q.len(),
            });
        }
        Ok(rotate_to_padded(&self.rotation, q))
    }
}

#[derive(Clone, Debug, Default)]
pub struct EncodeStats {
    pub total: usize,
    pub bytes_per_code: usize,
    pub total_bytes: usize,
}

/// Materialised multi-vector index keeping codes + norms side by side.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Index {
    pub quantizer: ExtendedRaBitQ,
    pub codes: Vec<Vec<u8>>,
    pub norms: Vec<f32>,
}

impl Index {
    pub fn new(quantizer: ExtendedRaBitQ) -> Self {
        Self {
            quantizer,
            codes: Vec::new(),
            norms: Vec::new(),
        }
    }

    pub fn add(&mut self, vectors: &[Vec<f32>]) -> Result<EncodeStats> {
        let mut stats = EncodeStats::default();
        for v in vectors {
            let (c, n) = self.quantizer.encode(v)?;
            stats.bytes_per_code = c.len();
            self.codes.push(c);
            self.norms.push(n);
            stats.total += 1;
        }
        stats.total_bytes = self.codes.iter().map(|c| c.len()).sum::<usize>()
            + self.norms.len() * std::mem::size_of::<f32>();
        Ok(stats)
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub(crate) fn pack_bits(codes: &[u8], bits: u8) -> Vec<u8> {
    let total_bits = codes.len() * bits as usize;
    let n_bytes = (total_bits + 7) / 8;
    let mut out = vec![0u8; n_bytes];
    for (i, c) in codes.iter().enumerate() {
        let bit_pos = i * bits as usize;
        let byte = bit_pos / 8;
        let off = bit_pos % 8;
        let val = (*c as u16) & ((1u16 << bits) - 1);
        out[byte] |= (val << off) as u8;
        let used = 8 - off;
        if (bits as usize) > used {
            out[byte + 1] |= (val >> used) as u8;
        }
    }
    out
}

pub(crate) fn unpack_bits(packed: &[u8], bits: u8, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mask = (1u16 << bits) - 1;
    for i in 0..n {
        let bit_pos = i * bits as usize;
        let byte = bit_pos / 8;
        let off = bit_pos % 8;
        let lo = packed[byte] as u16 >> off;
        let used = 8 - off;
        let full = if (bits as usize) > used && byte + 1 < packed.len() {
            lo | ((packed[byte + 1] as u16) << used)
        } else {
            lo
        };
        out.push((full & mask) as u8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn pack_unpack_roundtrip() {
        for bits in [1u8, 2, 3, 4, 6, 8] {
            let n = 97;
            let max = (1u8 << bits).saturating_sub(1);
            let mut rng = ChaCha8Rng::seed_from_u64(bits as u64);
            let codes: Vec<u8> = (0..n).map(|_| rng.gen_range(0..=max)).collect();
            let packed = pack_bits(&codes, bits);
            let back = unpack_bits(&packed, bits, n);
            assert_eq!(codes, back, "bits={}", bits);
        }
    }

    #[test]
    fn encode_decode_matches_dim() {
        let q = ExtendedRaBitQ::new(128, 4, 7).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let v: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
        let (codes, _n) = q.encode(&v).unwrap();
        let dec = q.decode_unit(&codes);
        assert_eq!(dec.len(), q.padded_dim);
    }
}
