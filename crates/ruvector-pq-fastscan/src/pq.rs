//! Standard 8-bit Product Quantization with scalar ADC scan.
//!
//! Used as the apples-to-apples baseline for FastScan. Codebook training is
//! shared (each subquantizer is just a slice with k centroids); only the
//! storage layout and scan kernel differ.

use crate::{kmeans, PqError, Result};

/// Trained product quantizer over `dim`-dimensional vectors, split into
/// `m` equal subspaces, each with `k` centroids.
///
/// Layout: `centroids[sub]` is a flat `k * dsub` Vec.
pub struct ProductQuantizer {
    pub dim: usize,
    pub m: usize,
    pub dsub: usize,
    pub k: usize,
    pub centroids: Vec<Vec<f32>>, // length m, each k * dsub
}

impl ProductQuantizer {
    pub fn train(
        train: &[f32],
        n: usize,
        dim: usize,
        m: usize,
        k: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self> {
        if dim % m != 0 {
            return Err(PqError::BadSubspace { dim, m });
        }
        if n == 0 {
            return Err(PqError::EmptyTraining);
        }
        if k > n {
            return Err(PqError::TooFewSamples { k, n });
        }
        if train.len() != n * dim {
            return Err(PqError::BadVectorCount {
                expected: n * dim,
                actual: train.len(),
            });
        }
        let dsub = dim / m;

        let mut centroids = Vec::with_capacity(m);
        for s in 0..m {
            // Extract subspace columns [s*dsub .. (s+1)*dsub] into a contiguous buffer.
            let mut sub = vec![0f32; n * dsub];
            for i in 0..n {
                let src = &train[i * dim + s * dsub..i * dim + (s + 1) * dsub];
                sub[i * dsub..(i + 1) * dsub].copy_from_slice(src);
            }
            let c = kmeans::kmeans(&sub, n, dsub, k, iters, seed.wrapping_add(s as u64));
            centroids.push(c);
        }

        Ok(Self { dim, m, dsub, k, centroids })
    }

    /// Encode `n` vectors. Returns `n * m` codes (row-major).
    /// Each code is the centroid index within its subquantizer.
    /// `k` must be <= 256.
    pub fn encode(&self, vectors: &[f32], n: usize) -> Vec<u8> {
        assert!(self.k <= 256);
        assert_eq!(vectors.len(), n * self.dim);
        let mut codes = vec![0u8; n * self.m];
        let mut sub = vec![0f32; n * self.dsub];
        for s in 0..self.m {
            for i in 0..n {
                let src = &vectors[i * self.dim + s * self.dsub..i * self.dim + (s + 1) * self.dsub];
                sub[i * self.dsub..(i + 1) * self.dsub].copy_from_slice(src);
            }
            let assigned = kmeans::assign_nearest(&sub, n, self.dsub, &self.centroids[s], self.k);
            for i in 0..n {
                codes[i * self.m + s] = assigned[i];
            }
        }
        codes
    }

    /// Build the per-query ADC distance table: `m * k` f32 distances.
    /// `table[s * k + c]` = squared L2 between query subvector `s` and centroid `c` of subspace `s`.
    pub fn build_lut_f32(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim);
        let mut lut = vec![0f32; self.m * self.k];
        for s in 0..self.m {
            let q = &query[s * self.dsub..(s + 1) * self.dsub];
            for c in 0..self.k {
                let cc = &self.centroids[s][c * self.dsub..(c + 1) * self.dsub];
                let mut acc = 0f32;
                for i in 0..self.dsub {
                    let d = q[i] - cc[i];
                    acc += d * d;
                }
                lut[s * self.k + c] = acc;
            }
        }
        lut
    }
}

/// Storage-and-scan wrapper for 8-bit PQ codes (row-major: `n * m` u8).
pub struct Pq8Index {
    pub pq: ProductQuantizer,
    pub codes: Vec<u8>,
    pub n: usize,
}

impl Pq8Index {
    pub fn from_vectors(pq: ProductQuantizer, vectors: &[f32], n: usize) -> Self {
        let codes = pq.encode(vectors, n);
        Self { pq, codes, n }
    }

    /// Scalar ADC scan. Returns top-`k` (idx, approx_sq_dist) sorted ascending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let lut = self.pq.build_lut_f32(query);
        let m = self.pq.m;
        let kc = self.pq.k;
        let mut out: Vec<(u32, f32)> = (0..self.n)
            .map(|i| {
                let row = &self.codes[i * m..(i + 1) * m];
                let mut acc = 0f32;
                for s in 0..m {
                    acc += lut[s * kc + row[s] as usize];
                }
                (i as u32, acc)
            })
            .collect();
        out.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out.truncate(k);
        out
    }
}
