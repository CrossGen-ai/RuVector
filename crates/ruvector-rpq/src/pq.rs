//! Product Quantizer (single-level), 8-bit codes per subspace.

use crate::kmeans::{kmeans, sq_l2};
use crate::rng::Rng;
use crate::Quantizer;

/// Product Quantizer: split `dim` into `m` equal subspaces, k-means each with
/// `k = 2^bits_per_sub` centroids. Only `bits_per_sub == 8` (k=256) supported
/// for the benchmark; the code path is a straightforward extension for other
/// bit widths.
pub struct Pq {
    pub(crate) dim: usize,
    pub(crate) m: usize,
    pub(crate) sub_dim: usize,
    pub(crate) k: usize,
    /// Codebooks: `m` blocks of `k * sub_dim` floats.
    pub(crate) codebooks: Vec<f32>,
}

impl Pq {
    /// Train a PQ on `train` (row-major, `n * dim` floats).
    pub fn train(train: &[f32], dim: usize, m: usize, iters: usize, rng: &mut Rng) -> Self {
        assert!(dim % m == 0, "dim ({dim}) not divisible by m ({m})");
        let sub_dim = dim / m;
        let k = 256; // fixed 8-bit codes per subspace
        let n = train.len() / dim;
        let mut codebooks = vec![0.0f32; m * k * sub_dim];
        // Scratch buffer for one subspace's training data.
        let mut sub = vec![0.0f32; n * sub_dim];
        for j in 0..m {
            // Gather column-block j into `sub` contiguously.
            for i in 0..n {
                let src = &train[i * dim + j * sub_dim..i * dim + (j + 1) * sub_dim];
                sub[i * sub_dim..(i + 1) * sub_dim].copy_from_slice(src);
            }
            let cb = kmeans(&sub, sub_dim, k, iters, rng);
            codebooks[j * k * sub_dim..(j + 1) * k * sub_dim].copy_from_slice(&cb);
        }
        Self { dim, m, sub_dim, k, codebooks }
    }

    /// Number of subspaces.
    pub fn m(&self) -> usize {
        self.m
    }
    /// Number of centroids per subspace (256 for 8-bit codes).
    pub fn k(&self) -> usize {
        self.k
    }
    /// Sub-vector dimensionality.
    pub fn sub_dim(&self) -> usize {
        self.sub_dim
    }
    /// Raw codebook slice for subspace `j` (`k * sub_dim` floats).
    pub fn codebook(&self, j: usize) -> &[f32] {
        &self.codebooks[j * self.k * self.sub_dim..(j + 1) * self.k * self.sub_dim]
    }

    /// Reconstruct approximate vector from its code (helper for RPQ residual step).
    pub fn reconstruct(&self, code: &[u8], out: &mut [f32]) {
        debug_assert_eq!(code.len(), self.m);
        debug_assert_eq!(out.len(), self.dim);
        for j in 0..self.m {
            let c = code[j] as usize;
            let src = &self.codebooks
                [j * self.k * self.sub_dim + c * self.sub_dim
                    ..j * self.k * self.sub_dim + (c + 1) * self.sub_dim];
            let dst = &mut out[j * self.sub_dim..(j + 1) * self.sub_dim];
            dst.copy_from_slice(src);
        }
    }

    /// Encode a single vector (produces `m` bytes).
    pub fn encode_raw(&self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.m);
        for j in 0..self.m {
            let sub_x = &x[j * self.sub_dim..(j + 1) * self.sub_dim];
            let cb = self.codebook(j);
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let d = sq_l2(sub_x, &cb[c * self.sub_dim..(c + 1) * self.sub_dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            out[j] = best as u8;
        }
    }

    /// Precompute per-subspace lookup table of shape [m][k] holding
    /// `sq_l2(query_sub_j, centroid_j_c)`.
    pub fn compute_lut(&self, query: &[f32]) -> Vec<f32> {
        debug_assert_eq!(query.len(), self.dim);
        let mut lut = vec![0.0f32; self.m * self.k];
        for j in 0..self.m {
            let q = &query[j * self.sub_dim..(j + 1) * self.sub_dim];
            let cb = self.codebook(j);
            for c in 0..self.k {
                lut[j * self.k + c] =
                    sq_l2(q, &cb[c * self.sub_dim..(c + 1) * self.sub_dim]);
            }
        }
        lut
    }

    /// Sum the LUT entries selected by `code`.
    #[inline]
    pub fn adc_from_lut(&self, lut: &[f32], code: &[u8]) -> f32 {
        let mut s = 0.0f32;
        for j in 0..self.m {
            s += lut[j * self.k + code[j] as usize];
        }
        s
    }
}

impl Quantizer for Pq {
    fn dim(&self) -> usize {
        self.dim
    }
    fn code_bytes(&self) -> usize {
        self.m
    }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        self.encode_raw(x, out);
    }
    fn adc_sq_distance(&self, query: &[f32], encoded: &[u8]) -> f32 {
        let lut = self.compute_lut(query);
        self.adc_from_lut(&lut, encoded)
    }
    fn name(&self) -> &'static str {
        "pq"
    }
}
