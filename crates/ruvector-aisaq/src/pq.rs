//! Minimal Product Quantizer.
//!
//! Splits each `D`-dim vector into `m` subvectors of `D/m` dims,
//! trains `k = 2^bits` centroids per subspace with mini-batch
//! k-means, and encodes each vector as `m` bytes (when `bits <= 8`).
//!
//! ADC (asymmetric distance computation) precomputes a
//! `m x k` table of squared distances between the query's
//! subvectors and each centroid; then a code's distance is a
//! sum of `m` table lookups.
//!
//! This is deliberately compact — enough to get real numbers,
//! not a production quantizer. The public trait boundary in
//! `backends.rs` means a smarter PQ (OPQ, RaBitQ, LVQ) can
//! swap in without disturbing the graph or the mmap code path.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::l2_sq;

/// A trained product quantizer with 8-bit codes (k = 256 per subspace).
pub struct ProductQuantizer {
    pub d: usize,
    pub m: usize,
    pub dsub: usize,
    /// `m` codebooks laid out contiguously; each codebook is `k * dsub` f32s.
    pub codebooks: Vec<f32>,
}

impl ProductQuantizer {
    pub const K: usize = 256;

    /// Train PQ codebooks with 25 iterations of Lloyd's algorithm per
    /// subspace on the provided training vectors.
    pub fn train(vectors: &[f32], d: usize, m: usize, seed: u64) -> Self {
        assert!(d % m == 0, "d must be divisible by m");
        let dsub = d / m;
        let n = vectors.len() / d;
        assert!(n >= Self::K, "need >= {} training vectors", Self::K);

        let mut codebooks = vec![0.0f32; m * Self::K * dsub];
        let mut rng = StdRng::seed_from_u64(seed);

        // Train each subspace independently
        for sub in 0..m {
            let cb_off = sub * Self::K * dsub;

            // Seed centroids with random distinct training vectors.
            for c in 0..Self::K {
                let idx = rng.gen_range(0..n);
                let src = idx * d + sub * dsub;
                let dst = cb_off + c * dsub;
                codebooks[dst..dst + dsub].copy_from_slice(&vectors[src..src + dsub]);
            }

            // Lloyd's iterations
            let mut assign = vec![0u16; n];
            let mut sums = vec![0.0f32; Self::K * dsub];
            let mut counts = vec![0u32; Self::K];

            for _iter in 0..25 {
                // Assign
                for i in 0..n {
                    let vsub = &vectors[i * d + sub * dsub..i * d + sub * dsub + dsub];
                    let mut best = 0u16;
                    let mut best_d = f32::INFINITY;
                    for c in 0..Self::K {
                        let cb = &codebooks[cb_off + c * dsub..cb_off + (c + 1) * dsub];
                        let dist = l2_sq(vsub, cb);
                        if dist < best_d {
                            best_d = dist;
                            best = c as u16;
                        }
                    }
                    assign[i] = best;
                }

                // Update
                for v in sums.iter_mut() { *v = 0.0; }
                for v in counts.iter_mut() { *v = 0; }
                for i in 0..n {
                    let c = assign[i] as usize;
                    counts[c] += 1;
                    let src = i * d + sub * dsub;
                    for j in 0..dsub {
                        sums[c * dsub + j] += vectors[src + j];
                    }
                }
                for c in 0..Self::K {
                    if counts[c] == 0 {
                        // Re-seed dead centroid from a random point.
                        let idx = rng.gen_range(0..n);
                        let src = idx * d + sub * dsub;
                        let dst = cb_off + c * dsub;
                        codebooks[dst..dst + dsub].copy_from_slice(&vectors[src..src + dsub]);
                    } else {
                        let inv = 1.0 / counts[c] as f32;
                        for j in 0..dsub {
                            codebooks[cb_off + c * dsub + j] = sums[c * dsub + j] * inv;
                        }
                    }
                }
            }
        }

        Self { d, m, dsub, codebooks }
    }

    /// Encode a single vector to `m` bytes.
    pub fn encode(&self, v: &[f32], out: &mut [u8]) {
        debug_assert_eq!(v.len(), self.d);
        debug_assert_eq!(out.len(), self.m);
        for sub in 0..self.m {
            let cb_off = sub * Self::K * self.dsub;
            let vsub = &v[sub * self.dsub..(sub + 1) * self.dsub];
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for c in 0..Self::K {
                let cb = &self.codebooks[cb_off + c * self.dsub..cb_off + (c + 1) * self.dsub];
                let dist = l2_sq(vsub, cb);
                if dist < best_d {
                    best_d = dist;
                    best = c as u8;
                }
            }
            out[sub] = best;
        }
    }

    /// Encode an entire dataset (row-major) into a flat `n * m` byte array.
    pub fn encode_all(&self, vectors: &[f32]) -> Vec<u8> {
        let n = vectors.len() / self.d;
        let mut codes = vec![0u8; n * self.m];
        for i in 0..n {
            let v = &vectors[i * self.d..(i + 1) * self.d];
            let out = &mut codes[i * self.m..(i + 1) * self.m];
            self.encode(v, out);
        }
        codes
    }

    /// Build the ADC lookup table for `query`: shape `m x K`, row-major.
    pub fn build_lut(&self, query: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.d);
        let mut lut = vec![0.0f32; self.m * Self::K];
        for sub in 0..self.m {
            let cb_off = sub * Self::K * self.dsub;
            let qsub = &query[sub * self.dsub..(sub + 1) * self.dsub];
            for c in 0..Self::K {
                let cb = &self.codebooks[cb_off + c * self.dsub..cb_off + (c + 1) * self.dsub];
                lut[sub * Self::K + c] = l2_sq(qsub, cb);
            }
        }
        lut
    }

    /// Sum ADC lookups for a `m`-byte code against a prebuilt `lut`.
    #[inline]
    pub fn adc(&self, code: &[u8], lut: &[f32]) -> f32 {
        debug_assert_eq!(code.len(), self.m);
        let mut acc = 0.0f32;
        for sub in 0..self.m {
            acc += lut[sub * Self::K + code[sub] as usize];
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gaussian_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut out = vec![0.0f32; n * d];
        for v in out.iter_mut() { *v = rng.gen_range(-1.0..1.0); }
        out
    }

    #[test]
    fn pq_reconstruction_bounded() {
        let d = 32;
        let m = 8;
        let data = gaussian_data(1024, d, 7);
        let pq = ProductQuantizer::train(&data, d, m, 7);
        assert_eq!(pq.codebooks.len(), m * ProductQuantizer::K * (d / m));
        let mut code = vec![0u8; m];
        pq.encode(&data[0..d], &mut code);
        let lut = pq.build_lut(&data[0..d]);
        let adc = pq.adc(&code, &lut);
        // Distance from a point to its own quantization should be small
        // relative to the average pairwise distance.
        assert!(adc < 10.0, "adc={adc}");
    }
}
