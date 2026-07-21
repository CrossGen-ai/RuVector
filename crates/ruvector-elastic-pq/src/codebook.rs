//! Per-subspace codebook trained by k-means.

use crate::kmeans::{kmeans, sq_l2};

/// A trained codebook for one PQ subspace.
#[derive(Clone)]
pub struct Codebook {
    /// Number of bits used to store one code index (1..=8 supported).
    pub bits: u8,
    /// Subvector dimensionality this codebook lives in.
    pub sub_dim: usize,
    /// `k * sub_dim` centroid buffer laid out row-major (k = 1 << bits).
    pub centroids: Vec<f32>,
    /// Training distortion (sum of assigned-centroid squared distances).
    pub train_distortion: f64,
}

impl Codebook {
    /// Number of centroids in this codebook (`1 << bits`).
    #[inline]
    pub fn k(&self) -> usize {
        1usize << self.bits as usize
    }

    /// Train a codebook of `bits` bits on the provided subvectors.
    pub fn train(subvectors: &[f32], n: usize, sub_dim: usize, bits: u8, seed: u64) -> Self {
        assert!(bits >= 1 && bits <= 8, "bits must be in 1..=8");
        let k = 1usize << bits as usize;
        let effective_k = k.min(n);
        let res = kmeans(subvectors, n, sub_dim, effective_k, 25, seed);
        // If we requested more codes than we had unique points, pad the
        // extra centroids by duplicating the last valid one — encode still
        // works, they'll simply never win an assignment.
        let mut centroids = res.centroids;
        if effective_k < k {
            let last = centroids[(effective_k - 1) * sub_dim..effective_k * sub_dim].to_vec();
            centroids.resize(k * sub_dim, 0.0);
            for c in effective_k..k {
                centroids[c * sub_dim..(c + 1) * sub_dim].copy_from_slice(&last);
            }
        }
        Codebook {
            bits,
            sub_dim,
            centroids,
            train_distortion: res.distortion,
        }
    }

    /// Encode a subvector to the index of the nearest centroid.
    #[inline]
    pub fn encode(&self, sub: &[f32]) -> u16 {
        debug_assert_eq!(sub.len(), self.sub_dim);
        let mut best = 0u16;
        let mut best_d = f32::INFINITY;
        for c in 0..self.k() {
            let d = sq_l2(sub, &self.centroids[c * self.sub_dim..(c + 1) * self.sub_dim]);
            if d < best_d {
                best_d = d;
                best = c as u16;
            }
        }
        best
    }

    /// Build the query-side lookup table used by ADC: for every centroid,
    /// pre-compute the squared L2 distance between the query subvector and
    /// that centroid. Length is `k()`.
    pub fn adc_table(&self, query_sub: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query_sub.len(), self.sub_dim);
        let k = self.k();
        let mut table = Vec::with_capacity(k);
        for c in 0..k {
            table.push(sq_l2(
                query_sub,
                &self.centroids[c * self.sub_dim..(c + 1) * self.sub_dim],
            ));
        }
        table
    }
}
