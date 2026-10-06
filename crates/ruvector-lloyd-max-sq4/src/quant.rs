//! Codec trait and two implementations: Uniform SQ4 and Lloyd-Max SQ4.
//!
//! Both store `dim` codes per vector as `(dim + 1) / 2` bytes (nibble packed).

use crate::train::fit_lloyd_max_1d;
use crate::{pack_nibble, unpack_nibble, NLEVELS};

/// A per-dimension scalar quantizer for 4-bit codes.
pub trait Codec: Send + Sync {
    /// Vector dimensionality.
    fn dim(&self) -> usize;
    /// Bytes needed to store one code vector.
    fn code_bytes(&self) -> usize {
        (self.dim() + 1) / 2
    }
    /// Encode a single float vector into a nibble-packed byte buffer.
    fn encode(&self, v: &[f32], out: &mut [u8]);
    /// Dequantize a code vector to float, allocating a Vec.
    fn decode(&self, code: &[u8]) -> Vec<f32> {
        let mut out = vec![0f32; self.dim()];
        self.decode_into(code, &mut out);
        out
    }
    /// Dequantize a code vector into a provided buffer.
    fn decode_into(&self, code: &[u8], out: &mut [f32]);
    /// Return the centroid value for dim `d`, level `k`.
    fn centroid(&self, d: usize, k: usize) -> f32;

    /// Batch encode `n_vectors` row-major vectors.
    fn encode_batch(&self, flat: &[f32], n: usize) -> Vec<u8> {
        assert_eq!(flat.len(), n * self.dim());
        let cb = self.code_bytes();
        let mut out = vec![0u8; n * cb];
        for i in 0..n {
            let v = &flat[i * self.dim()..(i + 1) * self.dim()];
            self.encode(v, &mut out[i * cb..(i + 1) * cb]);
        }
        out
    }
}

/// Uniform SQ4: linear bins per dimension between (min, max).
pub struct UniformSQ4 {
    dim: usize,
    mins: Vec<f32>,
    scales: Vec<f32>, // (max - min)
    centroids: Vec<[f32; NLEVELS]>,
}

impl UniformSQ4 {
    /// Fit on a training set `flat` of `n` vectors.
    pub fn fit(flat: &[f32], n: usize, dim: usize) -> Self {
        assert_eq!(flat.len(), n * dim);
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for i in 0..n {
            for d in 0..dim {
                let x = flat[i * dim + d];
                if x < mins[d] {
                    mins[d] = x;
                }
                if x > maxs[d] {
                    maxs[d] = x;
                }
            }
        }
        let mut scales = vec![0f32; dim];
        let mut centroids = Vec::with_capacity(dim);
        for d in 0..dim {
            let s = (maxs[d] - mins[d]).max(f32::EPSILON);
            scales[d] = s;
            let mut row = [0f32; NLEVELS];
            for k in 0..NLEVELS {
                row[k] = mins[d] + s * (k as f32 + 0.5) / NLEVELS as f32;
            }
            centroids.push(row);
        }
        Self {
            dim,
            mins,
            scales,
            centroids,
        }
    }
}

impl Codec for UniformSQ4 {
    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, v: &[f32], out: &mut [u8]) {
        assert_eq!(v.len(), self.dim);
        assert_eq!(out.len(), self.code_bytes());
        let mut i = 0;
        let mut pair = [0u8; 2];
        let mut have = 0;
        for d in 0..self.dim {
            let x = v[d];
            let n = NLEVELS as f32;
            // Normalize to [0, NLEVELS), clamp to [0, NLEVELS - 1].
            let norm = ((x - self.mins[d]) / self.scales[d]) * n;
            let k = norm.clamp(0.0, n - 1.0 + f32::EPSILON) as u8;
            let k = k.min(NLEVELS as u8 - 1);
            pair[have] = k;
            have += 1;
            if have == 2 {
                out[i] = pack_nibble(pair[0], pair[1]);
                i += 1;
                have = 0;
            }
        }
        if have == 1 {
            out[i] = pack_nibble(pair[0], 0);
        }
    }

    fn decode_into(&self, code: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        for d in 0..self.dim {
            let byte = code[d / 2];
            let (a, b) = unpack_nibble(byte);
            let k = if d % 2 == 0 { a } else { b };
            out[d] = self.centroids[d][k as usize];
        }
    }

    fn centroid(&self, d: usize, k: usize) -> f32 {
        self.centroids[d][k]
    }
}

/// Lloyd-Max 4-bit SQ: per-dimension optimized non-uniform bins.
pub struct LloydMaxSQ4 {
    dim: usize,
    centroids: Vec<[f32; NLEVELS]>, // sorted ascending per dim
}

impl LloydMaxSQ4 {
    /// Fit on a training set `flat` of `n` vectors.
    pub fn fit(flat: &[f32], n: usize, dim: usize, max_iters: usize, tol: f32) -> Self {
        assert_eq!(flat.len(), n * dim);
        let mut centroids = Vec::with_capacity(dim);
        let mut column = vec![0f32; n];
        for d in 0..dim {
            for i in 0..n {
                column[i] = flat[i * dim + d];
            }
            centroids.push(fit_lloyd_max_1d(&column, max_iters, tol));
        }
        Self { dim, centroids }
    }

    /// Encode one value to the nearest level via binary search over the
    /// 16 sorted centroids. 4 compares per call.
    #[inline(always)]
    fn encode_scalar(centroids: &[f32; NLEVELS], x: f32) -> u8 {
        // Binary search for position; nearest is argmin |x - c[k]|. For sorted
        // centroids, the nearest is either floor index i or i+1.
        let (mut lo, mut hi) = (0usize, NLEVELS - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if centroids[mid] < x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let i = lo;
        if i == 0 {
            0
        } else if (x - centroids[i - 1]).abs() <= (centroids[i] - x).abs() {
            (i - 1) as u8
        } else {
            i as u8
        }
    }
}

impl Codec for LloydMaxSQ4 {
    fn dim(&self) -> usize {
        self.dim
    }

    fn encode(&self, v: &[f32], out: &mut [u8]) {
        assert_eq!(v.len(), self.dim);
        assert_eq!(out.len(), self.code_bytes());
        let mut i = 0;
        let mut pair = [0u8; 2];
        let mut have = 0;
        for d in 0..self.dim {
            let k = Self::encode_scalar(&self.centroids[d], v[d]);
            pair[have] = k;
            have += 1;
            if have == 2 {
                out[i] = pack_nibble(pair[0], pair[1]);
                i += 1;
                have = 0;
            }
        }
        if have == 1 {
            out[i] = pack_nibble(pair[0], 0);
        }
    }

    fn decode_into(&self, code: &[u8], out: &mut [f32]) {
        assert_eq!(out.len(), self.dim);
        for d in 0..self.dim {
            let byte = code[d / 2];
            let (a, b) = unpack_nibble(byte);
            let k = if d % 2 == 0 { a } else { b };
            out[d] = self.centroids[d][k as usize];
        }
    }

    fn centroid(&self, d: usize, k: usize) -> f32 {
        self.centroids[d][k]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_data(n: usize, d: usize) -> Vec<f32> {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        (0..n * d)
            .map(|_| {
                let x: f32 = StandardNormal.sample(&mut rng);
                x
            })
            .collect()
    }

    #[test]
    fn roundtrip_uniform_small_dim() {
        let data = gaussian_data(500, 8);
        let codec = UniformSQ4::fit(&data, 500, 8);
        let codes = codec.encode_batch(&data, 500);
        let mut acc = 0f32;
        for i in 0..500 {
            let dec = codec.decode(&codes[i * codec.code_bytes()..(i + 1) * codec.code_bytes()]);
            for d in 0..8 {
                let e = dec[d] - data[i * 8 + d];
                acc += e * e;
            }
        }
        let mse = acc / (500.0 * 8.0);
        assert!(mse < 0.05, "uniform SQ4 MSE unexpectedly high: {}", mse);
    }

    #[test]
    fn lloyd_beats_uniform_mse_on_gaussian() {
        let d = 16;
        let n = 2000;
        let data = gaussian_data(n, d);
        let u = UniformSQ4::fit(&data, n, d);
        let l = LloydMaxSQ4::fit(&data, n, d, 25, 1e-5);
        let cb = u.code_bytes();
        let codes_u = u.encode_batch(&data, n);
        let codes_l = l.encode_batch(&data, n);
        let mut mse_u = 0f32;
        let mut mse_l = 0f32;
        let mut buf = vec![0f32; d];
        for i in 0..n {
            u.decode_into(&codes_u[i * cb..(i + 1) * cb], &mut buf);
            for j in 0..d {
                let e = buf[j] - data[i * d + j];
                mse_u += e * e;
            }
            l.decode_into(&codes_l[i * cb..(i + 1) * cb], &mut buf);
            for j in 0..d {
                let e = buf[j] - data[i * d + j];
                mse_l += e * e;
            }
        }
        mse_u /= (n * d) as f32;
        mse_l /= (n * d) as f32;
        assert!(
            mse_l < mse_u * 0.75,
            "lloyd should beat uniform on gaussian: {} vs {}",
            mse_l,
            mse_u
        );
    }

    #[test]
    fn code_bytes_matches_formula() {
        let codec = UniformSQ4::fit(&vec![0f32; 17], 1, 17);
        assert_eq!(codec.code_bytes(), 9); // ceil(17/2)
    }
}
