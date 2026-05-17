//! Product Quantization with deterministic k-means and Asymmetric Distance Tables.
//!
//! Memory math (per vector):
//!   raw f32:    d * 4 bytes
//!   PQ codes:   M  bytes   (each subspace stored as u8 with K <= 256)
//! Compression ratio = (d * 4) / M.
//!
//! Distance estimation (asymmetric):
//!   For query q, precompute `lut[m][c] = || q_m - codebook[m][c] ||^2`
//!   in O(M * K * (d/M)) = O(K * d). Then per-vector cost is `M` table
//!   lookups + adds — independent of d/M.

use crate::error::SymphonyError;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Encoded vectors. `codes` is laid out as `n * m` row-major u8.
pub struct PqCodes {
    pub n: usize,
    pub m: usize,
    pub codes: Vec<u8>,
}

impl PqCodes {
    #[inline]
    pub fn code(&self, i: usize) -> &[u8] {
        &self.codes[i * self.m..(i + 1) * self.m]
    }
}

pub struct ProductQuantizer {
    pub d: usize,
    pub m: usize,
    pub k: usize,       // codewords per subspace; <= 256
    pub ds: usize,      // = d / m
    /// Layout: `m * k * ds` row-major: [subspace][centroid][dim].
    pub codebooks: Vec<f32>,
}

impl ProductQuantizer {
    pub fn train(
        data: &[f32],
        d: usize,
        m: usize,
        k: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self, SymphonyError> {
        if d == 0 || m == 0 { return Err(SymphonyError::InvalidParam("d/m must be > 0")); }
        if d % m != 0 { return Err(SymphonyError::NotDivisible { d, m }); }
        if k == 0 || k > 256 { return Err(SymphonyError::InvalidParam("k must be in 1..=256")); }
        if data.is_empty() { return Err(SymphonyError::Empty); }
        if data.len() % d != 0 { return Err(SymphonyError::DimensionMismatch { expected: d, got: data.len() % d }); }

        let n = data.len() / d;
        let ds = d / m;
        let mut codebooks = vec![0.0_f32; m * k * ds];
        let mut rng = StdRng::seed_from_u64(seed);

        for sub in 0..m {
            // Initialize centroids by random sample without replacement.
            let mut picks: Vec<usize> = Vec::with_capacity(k);
            while picks.len() < k {
                let candidate = rng.gen_range(0..n);
                if !picks.contains(&candidate) { picks.push(candidate); }
                if picks.len() == n { break; } // n < k edge case
            }
            for c in 0..k {
                let pick = picks[c.min(picks.len() - 1)];
                let src = &data[pick * d + sub * ds..pick * d + sub * ds + ds];
                let dst = &mut codebooks[sub * k * ds + c * ds..sub * k * ds + (c + 1) * ds];
                dst.copy_from_slice(src);
            }

            // Lloyd iterations.
            let mut assign = vec![0u8; n];
            let mut sums = vec![0.0_f32; k * ds];
            let mut counts = vec![0u32; k];
            for _ in 0..iters {
                // Assign.
                for i in 0..n {
                    let v = &data[i * d + sub * ds..i * d + sub * ds + ds];
                    let mut best = 0u8;
                    let mut best_d = f32::INFINITY;
                    for c in 0..k {
                        let cw = &codebooks[sub * k * ds + c * ds..sub * k * ds + (c + 1) * ds];
                        let mut acc = 0.0_f32;
                        for t in 0..ds { let z = v[t] - cw[t]; acc += z * z; }
                        if acc < best_d { best_d = acc; best = c as u8; }
                    }
                    assign[i] = best;
                }
                // Update.
                for s in sums.iter_mut() { *s = 0.0; }
                for c in counts.iter_mut() { *c = 0; }
                for i in 0..n {
                    let c = assign[i] as usize;
                    counts[c] += 1;
                    let v = &data[i * d + sub * ds..i * d + sub * ds + ds];
                    let acc = &mut sums[c * ds..(c + 1) * ds];
                    for t in 0..ds { acc[t] += v[t]; }
                }
                for c in 0..k {
                    if counts[c] == 0 {
                        // Re-seed empty centroid from a random training point.
                        let pick = rng.gen_range(0..n);
                        let src = &data[pick * d + sub * ds..pick * d + sub * ds + ds];
                        let dst = &mut codebooks[sub * k * ds + c * ds..sub * k * ds + (c + 1) * ds];
                        dst.copy_from_slice(src);
                    } else {
                        let inv = 1.0 / counts[c] as f32;
                        let dst = &mut codebooks[sub * k * ds + c * ds..sub * k * ds + (c + 1) * ds];
                        let src = &sums[c * ds..(c + 1) * ds];
                        for t in 0..ds { dst[t] = src[t] * inv; }
                    }
                }
            }
        }

        Ok(Self { d, m, k, ds, codebooks })
    }

    pub fn encode_all(&self, data: &[f32]) -> Result<PqCodes, SymphonyError> {
        if data.len() % self.d != 0 {
            return Err(SymphonyError::DimensionMismatch { expected: self.d, got: data.len() % self.d });
        }
        let n = data.len() / self.d;
        let mut codes = vec![0u8; n * self.m];
        for i in 0..n {
            for sub in 0..self.m {
                let v = &data[i * self.d + sub * self.ds..i * self.d + sub * self.ds + self.ds];
                let mut best = 0u8;
                let mut best_d = f32::INFINITY;
                for c in 0..self.k {
                    let cw = &self.codebooks[sub * self.k * self.ds + c * self.ds..sub * self.k * self.ds + (c + 1) * self.ds];
                    let mut acc = 0.0_f32;
                    for t in 0..self.ds { let z = v[t] - cw[t]; acc += z * z; }
                    if acc < best_d { best_d = acc; best = c as u8; }
                }
                codes[i * self.m + sub] = best;
            }
        }
        Ok(PqCodes { n, m: self.m, codes })
    }

    /// Asymmetric distance table for a query. Shape `m * k`.
    pub fn build_adt(&self, query: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0_f32; self.m * self.k];
        for sub in 0..self.m {
            let q = &query[sub * self.ds..sub * self.ds + self.ds];
            for c in 0..self.k {
                let cw = &self.codebooks[sub * self.k * self.ds + c * self.ds..sub * self.k * self.ds + (c + 1) * self.ds];
                let mut acc = 0.0_f32;
                for t in 0..self.ds { let z = q[t] - cw[t]; acc += z * z; }
                lut[sub * self.k + c] = acc;
            }
        }
        lut
    }

    /// Estimated squared L2 from a query (via its ADT) to vector `i`.
    #[inline]
    pub fn adc_distance(&self, lut: &[f32], code: &[u8]) -> f32 {
        let mut acc = 0.0_f32;
        for sub in 0..self.m {
            acc += lut[sub * self.k + code[sub] as usize];
        }
        acc
    }

    /// Compression ratio versus raw f32 storage.
    pub fn compression_ratio(&self) -> f32 {
        (self.d * 4) as f32 / self.m as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn synth(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    #[test]
    fn train_and_encode_roundtrip() {
        let d = 16; let m = 4; let k = 16;
        let data = synth(200, d, 1);
        let pq = ProductQuantizer::train(&data, d, m, k, 10, 7).unwrap();
        let codes = pq.encode_all(&data).unwrap();
        assert_eq!(codes.codes.len(), 200 * m);
        // ADT distance is non-negative.
        let q = &data[0..d];
        let lut = pq.build_adt(q);
        let est = pq.adc_distance(&lut, codes.code(0));
        assert!(est >= 0.0);
    }
}
