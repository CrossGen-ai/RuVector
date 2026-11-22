//! Gray Code Scalar Quantization (GCSQ)
//!
//! Per-dimension uniform scalar quantization where bin indices are assigned via
//! the reflected binary Gray code. Because adjacent bins differ in exactly one
//! bit, a vector whose per-dimension bins drift by `k` steps changes exactly
//! `k` bits in the packed code (summed over dimensions) — making bit-packed
//! Hamming distance a tight, piecewise-linear proxy for L2 distance.
//!
//! The crate exposes a backend-swappable `Reranker` trait so callers can plug
//! in alternative short-code rerankers while keeping the ANN frontend fixed.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub mod backends;

pub use backends::{BinarySqReranker, GcsqReranker, SimHashReranker};

/// A uniform per-dimension scalar quantizer trained by scanning min/max.
///
/// `bits_per_dim` ∈ {1, 2, 4, 8}. `min[d]` / `max[d]` bound each dimension.
#[derive(Clone, Debug)]
pub struct ScalarRange {
    pub dim: usize,
    pub bits_per_dim: u8,
    pub min: Vec<f32>,
    pub scale: Vec<f32>, // levels / (max - min) per-dim; inf-safe (0 if constant)
}

impl ScalarRange {
    pub fn fit(train: &[Vec<f32>], bits_per_dim: u8) -> Self {
        assert!(matches!(bits_per_dim, 1 | 2 | 4 | 8));
        assert!(!train.is_empty());
        let dim = train[0].len();
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for v in train {
            for (d, x) in v.iter().enumerate() {
                if *x < lo[d] {
                    lo[d] = *x;
                }
                if *x > hi[d] {
                    hi[d] = *x;
                }
            }
        }
        let levels = (1u32 << bits_per_dim) as f32;
        let scale: Vec<f32> = (0..dim)
            .map(|d| {
                let span = hi[d] - lo[d];
                if span > 1e-12 { levels / span } else { 0.0 }
            })
            .collect();
        Self {
            dim,
            bits_per_dim,
            min: lo,
            scale,
        }
    }

    /// Returns an integer bin for each dimension, in [0, 2^bits_per_dim).
    pub fn bins(&self, v: &[f32]) -> Vec<u8> {
        let levels = (1u32 << self.bits_per_dim) as f32;
        let last = levels as u32 - 1;
        v.iter()
            .enumerate()
            .map(|(d, x)| {
                let t = ((*x - self.min[d]) * self.scale[d]).floor();
                t.clamp(0.0, (last) as f32) as u8
            })
            .collect()
    }
}

/// Reflected-binary Gray code map: `n -> n XOR (n >> 1)`.
#[inline(always)]
pub fn gray_encode(n: u32) -> u32 {
    n ^ (n >> 1)
}

/// Inverse Gray code map.
#[inline(always)]
pub fn gray_decode(mut g: u32) -> u32 {
    let mut n = g;
    while {
        g >>= 1;
        g != 0
    } {
        n ^= g;
    }
    n
}

/// Pack per-dimension bin indices into a compact byte stream.
///
/// `use_gray`: true applies Gray encoding to each bin, false uses plain binary.
pub fn pack_bins(bins: &[u8], bits_per_dim: u8, use_gray: bool) -> Vec<u8> {
    let total_bits = bins.len() * bits_per_dim as usize;
    let out_len = (total_bits + 7) / 8;
    let mut out = vec![0u8; out_len];
    let mask = (1u32 << bits_per_dim) - 1;
    for (i, &b) in bins.iter().enumerate() {
        let v = if use_gray {
            gray_encode(b as u32) & mask
        } else {
            b as u32 & mask
        };
        let bit = i * bits_per_dim as usize;
        let byte = bit / 8;
        let shift = bit % 8;
        // Little-endian within code; handle crossings for bits_per_dim <= 8.
        let hi = shift + bits_per_dim as usize;
        if hi <= 8 {
            out[byte] |= (v as u8) << shift;
        } else {
            let low_bits = 8 - shift;
            out[byte] |= ((v & ((1 << low_bits) - 1)) as u8) << shift;
            out[byte + 1] |= (v >> low_bits) as u8;
        }
    }
    out
}

/// Fast population count sum over a code (Hamming weight of XOR is caller-side).
#[inline(always)]
pub fn popcount_xor(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0u32;
    // Scalar; portable. Rustc autovectorizes popcnt where supported.
    let mut i = 0;
    while i + 8 <= a.len() {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap())
            ^ u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        s += x.count_ones();
        i += 8;
    }
    while i < a.len() {
        s += (a[i] ^ b[i]).count_ones();
        i += 1;
    }
    s
}

/// Deterministic Gaussian sampler used by the SimHash baseline.
pub fn gaussian_matrix(seed: u64, rows: usize, cols: usize) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        // Box-Muller.
        let u1: f32 = rng.gen_range(1e-9f32..1.0);
        let u2: f32 = rng.gen_range(0.0f32..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let z = r * (2.0 * std::f32::consts::PI * u2).cos();
        out.push(z);
    }
    out
}

/// Exact L2 squared distance between two float vectors.
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = *x - *y;
        s += d * d;
    }
    s
}
