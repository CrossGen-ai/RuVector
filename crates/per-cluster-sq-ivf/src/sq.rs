//! 4-bit scalar quantization utilities.
//!
//! Codes are packed two per byte (little nibble = even index).
//! Encoding: code = round(clamp((x - min) / (max - min), 0, 1) * 15).
//! Decoding: x_hat = min + (code / 15) * (max - min).

#[inline]
pub fn encode_nibble(x: f32, lo: f32, hi: f32) -> u8 {
    if hi <= lo { return 0; }
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    (t * 15.0 + 0.5) as u8 & 0x0F
}

#[inline]
pub fn decode_nibble(c: u8, lo: f32, hi: f32) -> f32 {
    lo + (c as f32) * (1.0 / 15.0) * (hi - lo)
}

/// Encode a d-dim vector into ceil(d/2) bytes using per-dim (lo, hi).
pub fn encode_vec(v: &[f32], lo: &[f32], hi: &[f32], out: &mut [u8]) {
    let d = v.len();
    debug_assert_eq!(lo.len(), d);
    debug_assert_eq!(hi.len(), d);
    debug_assert_eq!(out.len(), (d + 1) / 2);
    for j in 0..d {
        let c = encode_nibble(v[j], lo[j], hi[j]);
        let byte_idx = j / 2;
        if j & 1 == 0 {
            out[byte_idx] = c;
        } else {
            out[byte_idx] |= c << 4;
        }
    }
}

/// Compute squared L2 between query and a packed 4-bit vector, decoding on the fly.
#[inline]
pub fn l2_sq_decoded(packed: &[u8], lo: &[f32], hi: &[f32], query: &[f32]) -> f32 {
    let d = query.len();
    debug_assert_eq!(packed.len(), (d + 1) / 2);
    let mut s = 0.0f32;
    for j in 0..d {
        let byte_idx = j / 2;
        let c = if j & 1 == 0 {
            packed[byte_idx] & 0x0F
        } else {
            packed[byte_idx] >> 4
        };
        let x_hat = decode_nibble(c, lo[j], hi[j]);
        let diff = x_hat - query[j];
        s += diff * diff;
    }
    s
}

/// Per-dimension (min, max) over a column view.
pub fn fit_ranges(points: &[f32], d: usize) -> (Vec<f32>, Vec<f32>) {
    let n = points.len() / d;
    let mut lo = vec![f32::INFINITY; d];
    let mut hi = vec![f32::NEG_INFINITY; d];
    for i in 0..n {
        for j in 0..d {
            let v = points[i * d + j];
            if v < lo[j] { lo[j] = v; }
            if v > hi[j] { hi[j] = v; }
        }
    }
    // Guard: if cluster has 1 point or constant dim, hi must be strictly > lo
    // to avoid div-by-zero later. Expand a hair.
    for j in 0..d {
        if !lo[j].is_finite() { lo[j] = 0.0; }
        if !hi[j].is_finite() { hi[j] = 0.0; }
        if hi[j] <= lo[j] { hi[j] = lo[j] + 1e-6; }
    }
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_bounds() {
        let v = [0.0f32, 0.5, 1.0, -1.0];
        let lo = [-1.0f32, 0.0, 0.0, -1.0];
        let hi = [1.0f32, 1.0, 2.0, 0.0];
        let mut packed = vec![0u8; 2];
        encode_vec(&v, &lo, &hi, &mut packed);
        let dec: Vec<f32> = (0..4).map(|j| {
            let c = if j & 1 == 0 { packed[j / 2] & 0x0F } else { packed[j / 2] >> 4 };
            decode_nibble(c, lo[j], hi[j])
        }).collect();
        // Max per-dim error is (hi-lo)/30.
        for j in 0..4 {
            let err = (dec[j] - v[j]).abs();
            let step = (hi[j] - lo[j]) / 15.0;
            assert!(err <= step * 0.5001, "j={} err={} step={}", j, err, step);
        }
    }

    #[test]
    fn fit_ranges_handles_constant_dim() {
        let data = vec![1.0f32, 2.0, 1.0, 2.0, 1.0, 2.0];
        let (lo, hi) = fit_ranges(&data, 2);
        assert_eq!(lo, vec![1.0, 2.0]);
        assert!(hi[0] > lo[0] && hi[1] > lo[1]);
    }

    #[test]
    fn encode_matches_decode() {
        // dense 4-bit should preserve order between quantized distances
        let q = vec![0.3f32, 0.7, 0.2, 0.9];
        let lo = vec![0.0f32; 4];
        let hi = vec![1.0f32; 4];
        let mut packed = vec![0u8; 2];
        encode_vec(&q, &lo, &hi, &mut packed);
        let d = l2_sq_decoded(&packed, &lo, &hi, &q);
        // self-distance should be small
        assert!(d < 0.01, "d={}", d);
    }
}
