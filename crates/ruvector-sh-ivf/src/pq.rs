//! Product Quantizer with k=256 codebooks and prefix-aware ADC.
//!
//! Each vector is split into M equal-size subspaces (dim d_sub = d/M).
//! Each subspace trains a k-means with K codebook entries. A vector encodes
//! to M bytes. For a query q, we build a per-subspace LUT of size M*K giving
//! squared distance from q's subvector to each codebook centroid. ADC sums
//! M lookups; the prefix variant sums only the first `m_prefix` of them for
//! a cheaper (lower-bound-in-expectation) estimate.

use crate::kmeans::Kmeans;

pub struct ProductQuantizer {
    pub d: usize,
    pub m: usize,
    pub k: usize,
    pub d_sub: usize,
    /// Codebooks laid out as m sub-codebooks of (k * d_sub) floats.
    pub codebooks: Vec<Vec<f32>>,
}

impl ProductQuantizer {
    pub fn train(base: &[f32], d: usize, m: usize, k: usize, seed: u64) -> Self {
        assert!(d % m == 0, "d must be divisible by m");
        let d_sub = d / m;
        let n = base.len() / d;

        // Gather each subspace into a dense n*d_sub slab, then k-means it.
        let mut codebooks = Vec::with_capacity(m);
        for sub in 0..m {
            let mut slab = vec![0f32; n * d_sub];
            for i in 0..n {
                for j in 0..d_sub {
                    slab[i * d_sub + j] = base[i * d + sub * d_sub + j];
                }
            }
            let km = Kmeans::train(&slab, d_sub, k, 12, seed.wrapping_add(sub as u64));
            codebooks.push(km.centroids);
        }
        ProductQuantizer {
            d,
            m,
            k,
            d_sub,
            codebooks,
        }
    }

    /// Encode a d-dim vector into M bytes.
    pub fn encode(&self, v: &[f32]) -> Vec<u8> {
        debug_assert_eq!(v.len(), self.d);
        let mut code = vec![0u8; self.m];
        for sub in 0..self.m {
            let x = &v[sub * self.d_sub..(sub + 1) * self.d_sub];
            let cb = &self.codebooks[sub];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let cent = &cb[c * self.d_sub..(c + 1) * self.d_sub];
                let mut s = 0f32;
                for j in 0..self.d_sub {
                    let diff = x[j] - cent[j];
                    s += diff * diff;
                }
                if s < best_d {
                    best_d = s;
                    best = c;
                }
            }
            code[sub] = best as u8;
        }
        code
    }

    /// Build the per-query ADC LUT, length m * k.
    pub fn build_lut(&self, q: &[f32]) -> Vec<f32> {
        debug_assert_eq!(q.len(), self.d);
        let mut lut = vec![0f32; self.m * self.k];
        for sub in 0..self.m {
            let qsub = &q[sub * self.d_sub..(sub + 1) * self.d_sub];
            let cb = &self.codebooks[sub];
            for c in 0..self.k {
                let cent = &cb[c * self.d_sub..(c + 1) * self.d_sub];
                let mut s = 0f32;
                for j in 0..self.d_sub {
                    let diff = qsub[j] - cent[j];
                    s += diff * diff;
                }
                lut[sub * self.k + c] = s;
            }
        }
        lut
    }

    /// Full ADC: sum M lookups.
    #[inline]
    pub fn adc_full(&self, code: &[u8], lut: &[f32]) -> f32 {
        let mut s = 0f32;
        for sub in 0..self.m {
            s += lut[sub * self.k + code[sub] as usize];
        }
        s
    }

    /// Prefix ADC: sum first `m_prefix` subspaces only.
    #[inline]
    pub fn adc_prefix(&self, code: &[u8], lut: &[f32], m_prefix: usize) -> f32 {
        let mp = m_prefix.min(self.m);
        let mut s = 0f32;
        for sub in 0..mp {
            s += lut[sub * self.k + code[sub] as usize];
        }
        s
    }
}
