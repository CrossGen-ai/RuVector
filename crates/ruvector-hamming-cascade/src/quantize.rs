//! Vector quantization primitives shared by the oracles.
//!
//! These functions are intentionally straightforward — the point of the
//! benchmark is to show what portable, allocation-free scalar code buys.

/// Quantize a single vector into per-vector INT8 with an associated
/// (min, max) affine dequant window. Returns the byte payload.
pub fn quantize_int8(vec: &[f32]) -> (Vec<u8>, f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in vec {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        lo = 0.0;
        hi = 0.0;
    }
    let range = (hi - lo).max(f32::EPSILON);
    let mut out = Vec::with_capacity(vec.len());
    for &v in vec {
        let n = ((v - lo) / range * 255.0).round().clamp(0.0, 255.0);
        out.push(n as u8);
    }
    (out, lo, hi)
}

/// Dequantize an INT8 byte into an approximate float given (lo, hi).
#[inline]
pub fn dequant_int8(b: u8, lo: f32, hi: f32) -> f32 {
    let range = (hi - lo).max(f32::EPSILON);
    lo + (b as f32) * range / 255.0
}

/// Sign-based 1-bit quantization. Each dimension is 1 if `v >= threshold`
/// else 0. The default threshold is the componentwise mean of the training
/// set (passed via `threshold_dim`).
pub fn quantize_binary(vec: &[f32], threshold_dim: &[f32]) -> Vec<u64> {
    debug_assert_eq!(vec.len(), threshold_dim.len());
    let words = vec.len().div_ceil(64);
    let mut out = vec![0u64; words];
    for (i, (&v, &t)) in vec.iter().zip(threshold_dim.iter()).enumerate() {
        if v >= t {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

/// Compute per-dimension mean thresholds from a training slice.
pub fn dim_means(vectors: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let mut means = vec![0.0f32; dim];
    if vectors.is_empty() {
        return means;
    }
    for v in vectors {
        for (m, &x) in means.iter_mut().zip(v.iter()) {
            *m += x;
        }
    }
    let n = vectors.len() as f32;
    for m in means.iter_mut() {
        *m /= n;
    }
    means
}

/// Popcount-based Hamming distance across `u64` words.
#[inline]
pub fn hamming_u64(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc: u32 = 0;
    for i in 0..a.len() {
        acc += (a[i] ^ b[i]).count_ones();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_roundtrip_bounded_error() {
        let v = vec![-1.0, -0.5, 0.0, 0.25, 1.0];
        let (bytes, lo, hi) = quantize_int8(&v);
        let recon: Vec<f32> = bytes.iter().map(|&b| dequant_int8(b, lo, hi)).collect();
        for (o, r) in v.iter().zip(recon.iter()) {
            assert!((o - r).abs() < (hi - lo) / 200.0, "err too big: {o} vs {r}");
        }
    }

    #[test]
    fn hamming_identity_zero() {
        let a = vec![0xDEAD_BEEF_1234_5678u64; 4];
        assert_eq!(hamming_u64(&a, &a), 0);
    }

    #[test]
    fn hamming_flip_all_bits() {
        let a = vec![0u64; 2];
        let b = vec![u64::MAX; 2];
        assert_eq!(hamming_u64(&a, &b), 128);
    }

    #[test]
    fn binary_quant_threshold_semantics() {
        let v = vec![-1.0, 0.0, 1.0, 2.0];
        let t = vec![0.0, 0.0, 0.0, 0.0];
        let q = quantize_binary(&v, &t);
        // bits: 0 (v[0]<0), 1 (v[1]>=0), 1, 1
        assert_eq!(q[0] & 0xF, 0b1110);
    }
}
