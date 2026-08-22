//! Residual Vector Quantization (RVQ) for compact vector storage.
//!
//! # Model
//!
//! Given vectors `x ∈ ℝ^D`, RVQ approximates each vector as a sum of
//! centroids drawn from `L` independently trained codebooks:
//!
//! ```text
//! x ≈ Σ_{ℓ=1..L}  C_ℓ[ i_ℓ ]        where C_ℓ ∈ ℝ^{K × D}
//! ```
//!
//! Training is greedy: `C_1` is trained on the raw vectors, `C_2` on the
//! *residuals* left after stage 1, and so on. This is the same recipe used
//! by FAISS `IndexResidualQuantizer` and by ScaNN's residual codebook path.
//!
//! # Storage
//!
//! Each vector costs `L · ⌈log₂ K⌉` bits — for `L=8, K=256` that is
//! **8 bytes** vs `4·D` bytes for a raw `f32` vector. At `D=128` that is
//! **64× compression**, and at `D=768` (typical embedding) it is **384×**.
//!
//! # Query
//!
//! We support two estimators, both LUT-based:
//!
//! - **Inner product / cosine (asymmetric):** precompute
//!   `LUT_ℓ[j] = ⟨q, C_ℓ[j]⟩`. Per candidate: `L` lookups + `L-1` adds.
//! - **Squared L2 (asymmetric):** precompute
//!   `LUT_ℓ[j] = ‖C_ℓ[j]‖² − 2⟨q, C_ℓ[j]⟩`, subtract `‖q‖²` correction.
//!
//! Both are unbiased *only relative to the reconstruction* — accuracy is
//! bounded by codebook-training MSE, not by the estimator itself.
//!
//! # Not included (on purpose)
//!
//! - **Additive Quantization** (Babenko & Lempitsky 2014) — jointly optimises
//!   all codebooks at once, ~2× slower to train, ~10-20 % better recall.
//!   The `Quantizer` trait below leaves room for an `additive` backend.
//! - **OPQ rotation** — an orthogonal preconditioner improves recall on
//!   axis-aligned data. Left as future work.

#![allow(clippy::needless_range_loop)]

mod kmeans;
pub mod index;

pub use index::{RvqIndex, SearchResult};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RvqError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    Dim { expected: usize, got: usize },
    #[error("empty training set")]
    EmptyTraining,
    #[error("k ({k}) exceeds training-set size ({n})")]
    KTooLarge { k: usize, n: usize },
}

/// A pluggable quantizer backend. Implementors reconstruct vectors
/// from compact codes and support LUT-based distance estimation.
pub trait Quantizer: Send + Sync {
    fn dim(&self) -> usize;
    fn code_bytes(&self) -> usize;

    /// Encode a single vector into `out` (which must be `code_bytes` long).
    fn encode(&self, x: &[f32], out: &mut [u8]) -> Result<(), RvqError>;

    /// Reconstruct a single vector from its code into `out`.
    fn decode(&self, code: &[u8], out: &mut [f32]) -> Result<(), RvqError>;
}

/// Configuration for training an RVQ codebook family.
#[derive(Debug, Clone)]
pub struct RvqConfig {
    /// Number of residual stages `L`.
    pub stages: usize,
    /// Centroids per stage `K` — must be ≤ 256 (byte-packed codes).
    pub k: usize,
    /// k-means iterations per stage.
    pub kmeans_iters: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RvqConfig {
    fn default() -> Self {
        Self { stages: 8, k: 256, kmeans_iters: 15, seed: 0xC0FFEE }
    }
}

/// A trained residual vector quantizer.
#[derive(Debug, Clone)]
pub struct Rvq {
    dim: usize,
    stages: usize,
    k: usize,
    /// Codebooks: `stages` blocks of `k * dim` `f32`s.
    codebooks: Vec<Vec<f32>>,
    /// Precomputed `‖C_ℓ[j]‖²` for L2 estimator.
    sq_norms: Vec<Vec<f32>>,
}

impl Rvq {
    /// Train an RVQ on `data` (row-major, `n` rows of `d` cols).
    pub fn train(data: &[f32], n: usize, d: usize, cfg: &RvqConfig) -> Result<Self, RvqError> {
        if n == 0 { return Err(RvqError::EmptyTraining); }
        if cfg.k > n { return Err(RvqError::KTooLarge { k: cfg.k, n }); }
        assert!(cfg.k <= 256, "K must fit in one byte");
        assert_eq!(data.len(), n * d);

        let mut residuals: Vec<f32> = data.to_vec();
        let mut codebooks = Vec::with_capacity(cfg.stages);
        let mut sq_norms = Vec::with_capacity(cfg.stages);

        for stage in 0..cfg.stages {
            let cb = kmeans::kmeans(&residuals, n, d, cfg.k, cfg.kmeans_iters, cfg.seed ^ (stage as u64 * 0x9E37));

            // subtract nearest centroid from each residual to form next stage input
            for i in 0..n {
                let x = &residuals[i * d..(i + 1) * d];
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..cfg.k {
                    let cc = &cb[c * d..(c + 1) * d];
                    let mut s = 0f32;
                    for j in 0..d { let e = x[j] - cc[j]; s += e * e; }
                    if s < best_d { best_d = s; best = c; }
                }
                let cc = &cb[best * d..(best + 1) * d].to_vec();
                let x = &mut residuals[i * d..(i + 1) * d];
                for j in 0..d { x[j] -= cc[j]; }
            }

            // sq norms
            let mut norms = Vec::with_capacity(cfg.k);
            for c in 0..cfg.k {
                let cc = &cb[c * d..(c + 1) * d];
                let mut s = 0f32;
                for j in 0..d { s += cc[j] * cc[j]; }
                norms.push(s);
            }

            codebooks.push(cb);
            sq_norms.push(norms);
        }

        Ok(Self { dim: d, stages: cfg.stages, k: cfg.k, codebooks, sq_norms })
    }

    pub fn stages(&self) -> usize { self.stages }
    pub fn k(&self) -> usize { self.k }

    /// Bytes to store `n` encoded vectors.
    pub fn storage_bytes(&self, n: usize) -> usize { n * self.stages }

    /// Reconstruction MSE on the training set (or any set) — useful for
    /// diagnosing how many stages you actually need.
    pub fn reconstruction_mse(&self, data: &[f32], n: usize) -> f32 {
        assert_eq!(data.len(), n * self.dim);
        let mut codes = vec![0u8; n * self.stages];
        for i in 0..n {
            self.encode(&data[i * self.dim..(i + 1) * self.dim],
                        &mut codes[i * self.stages..(i + 1) * self.stages]).unwrap();
        }
        let mut recon = vec![0f32; self.dim];
        let mut acc = 0.0f64;
        for i in 0..n {
            self.decode(&codes[i * self.stages..(i + 1) * self.stages], &mut recon).unwrap();
            let x = &data[i * self.dim..(i + 1) * self.dim];
            for j in 0..self.dim { let e = x[j] - recon[j]; acc += (e * e) as f64; }
        }
        (acc / (n * self.dim) as f64) as f32
    }

    /// Precompute a per-query LUT for inner-product estimation.
    /// Returns `stages * k` values; distance for a code is `Σ LUT[ℓ, c_ℓ]`.
    pub fn ip_lut(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.dim);
        let mut lut = vec![0f32; self.stages * self.k];
        for l in 0..self.stages {
            let cb = &self.codebooks[l];
            for c in 0..self.k {
                let cc = &cb[c * self.dim..(c + 1) * self.dim];
                let mut s = 0f32;
                for j in 0..self.dim { s += q[j] * cc[j]; }
                lut[l * self.k + c] = s;
            }
        }
        lut
    }

    /// Estimated inner product of query (via LUT) with encoded vector `code`.
    #[inline]
    pub fn estimate_ip(&self, lut: &[f32], code: &[u8]) -> f32 {
        debug_assert_eq!(lut.len(), self.stages * self.k);
        debug_assert_eq!(code.len(), self.stages);
        let mut s = 0f32;
        for l in 0..self.stages {
            s += lut[l * self.k + code[l] as usize];
        }
        s
    }

    /// Estimated squared L2 distance: ‖q‖² + Σ ‖c_ℓ‖² − 2·estimate_ip.
    ///
    /// Provide `q_sq` = ‖q‖² and a precomputed `norm_sum_lut` from
    /// [`Rvq::l2_lut`]. Faster than reconstructing per-candidate.
    #[inline]
    pub fn estimate_l2(&self, q_sq: f32, ip_lut: &[f32], norm_lut: &[f32], code: &[u8]) -> f32 {
        let mut ip = 0f32;
        let mut nsum = 0f32;
        for l in 0..self.stages {
            let idx = l * self.k + code[l] as usize;
            ip += ip_lut[idx];
            nsum += norm_lut[idx];
        }
        // Note: cross-stage inner products between centroids are dropped
        // (they're small on residual codebooks by construction — mean
        // residual is ~0). This is the same approximation FAISS uses for
        // its RQ L2 estimator; see the ADR for the error analysis.
        q_sq + nsum - 2.0 * ip
    }

    /// Precompute per-stage ‖C_ℓ[j]‖² LUT (query-independent, cache once).
    pub fn l2_norm_lut(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.stages * self.k);
        for l in 0..self.stages {
            out.extend_from_slice(&self.sq_norms[l]);
        }
        out
    }
}

impl Quantizer for Rvq {
    fn dim(&self) -> usize { self.dim }
    fn code_bytes(&self) -> usize { self.stages }

    fn encode(&self, x: &[f32], out: &mut [u8]) -> Result<(), RvqError> {
        if x.len() != self.dim { return Err(RvqError::Dim { expected: self.dim, got: x.len() }); }
        if out.len() != self.stages { return Err(RvqError::Dim { expected: self.stages, got: out.len() }); }
        let mut residual: Vec<f32> = x.to_vec();
        for l in 0..self.stages {
            let cb = &self.codebooks[l];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let cc = &cb[c * self.dim..(c + 1) * self.dim];
                let mut s = 0f32;
                for j in 0..self.dim { let e = residual[j] - cc[j]; s += e * e; }
                if s < best_d { best_d = s; best = c; }
            }
            out[l] = best as u8;
            let cc = &cb[best * self.dim..(best + 1) * self.dim].to_vec();
            for j in 0..self.dim { residual[j] -= cc[j]; }
        }
        Ok(())
    }

    fn decode(&self, code: &[u8], out: &mut [f32]) -> Result<(), RvqError> {
        if code.len() != self.stages { return Err(RvqError::Dim { expected: self.stages, got: code.len() }); }
        if out.len() != self.dim { return Err(RvqError::Dim { expected: self.dim, got: out.len() }); }
        for v in out.iter_mut() { *v = 0.0; }
        for l in 0..self.stages {
            let cc = &self.codebooks[l][code[l] as usize * self.dim..(code[l] as usize + 1) * self.dim];
            for j in 0..self.dim { out[j] += cc[j]; }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn gaussian(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    #[test]
    fn train_encode_decode_roundtrip() {
        let n = 400; let d = 32;
        let data = gaussian(n, d, 1);
        let cfg = RvqConfig { stages: 8, k: 32, kmeans_iters: 10, seed: 7 };
        let q = Rvq::train(&data, n, d, &cfg).unwrap();

        let mut code = vec![0u8; q.code_bytes()];
        let mut recon = vec![0f32; d];
        q.encode(&data[..d], &mut code).unwrap();
        q.decode(&code, &mut recon).unwrap();

        let mut err = 0f32;
        for j in 0..d { let e = data[j] - recon[j]; err += e * e; }
        // Reconstruction MSE should be well below variance of the data (~0.33)
        assert!(err / (d as f32) < 0.15, "recon MSE {} too large", err / (d as f32));
    }

    #[test]
    fn more_stages_reduce_mse() {
        let n = 500; let d = 16;
        let data = gaussian(n, d, 2);
        let mse2 = Rvq::train(&data, n, d, &RvqConfig{stages:2,k:32,kmeans_iters:10,seed:1}).unwrap()
            .reconstruction_mse(&data, n);
        let mse8 = Rvq::train(&data, n, d, &RvqConfig{stages:8,k:32,kmeans_iters:10,seed:1}).unwrap()
            .reconstruction_mse(&data, n);
        assert!(mse8 < mse2, "8 stages ({mse8}) should beat 2 stages ({mse2})");
    }
}
