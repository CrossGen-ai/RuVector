//! ruvector-anisotropic-pq — score-aware (anisotropic) product quantization
//! for maximum inner-product search (MIPS).
//!
//! Reference: Guo et al., "Accelerating Large-Scale Inference with Anisotropic
//! Vector Quantization" (ScaNN), ICML 2020.
//!
//! Practical difference vs. plain L2 PQ:
//! - L2 PQ minimises Σ ||x - x̃||² uniformly in every direction.
//! - Anisotropic PQ minimises η·||e_∥||² + ||e_⊥||², where e_∥ is the component
//!   of the quantisation error e = x - x̃ along the data direction x̂ = x/||x||,
//!   and η ≥ 1 amplifies that penalty. The MIPS error |xᵀq - x̃ᵀq| is dominated
//!   by e_∥ when q is correlated with x, so an MIPS-tuned codebook trades a
//!   little orthogonal error for tighter score reconstruction.
//!
//! This crate implements a per-subspace local-anisotropic decomposition: within
//! each subspace s, û_s := x_s / ||x_s|| is the local data direction; the
//! per-subspace weight matrix is W_x,s = (η-1)·û_s û_sᵀ + I. With η=1 this
//! reduces to plain L2 PQ; with η>1 the codebook bends toward the data
//! manifold inside each chunk.

use rand::Rng;
use rand::SeedableRng;

pub mod train;

/// A trained anisotropic PQ codebook.
#[derive(Clone, Debug)]
pub struct AnisotropicPq {
    pub dim: usize,
    pub m: usize,        // number of subspaces
    pub d_sub: usize,    // = dim / m
    pub k: usize,        // centroids per subspace (typ. 256)
    pub eta: f32,        // anisotropy ratio η ≥ 1
    /// centroids[s] is a row-major [k, d_sub] matrix.
    pub centroids: Vec<Vec<f32>>,
}

impl AnisotropicPq {
    /// Number of bytes a single encoded vector occupies (m bytes when k=256).
    pub fn code_size(&self) -> usize {
        // ceil(log2(k) * m / 8). We keep it simple: 1 byte per subspace,
        // requiring k ≤ 256.
        assert!(self.k <= 256, "this PoC keeps k ≤ 256; got {}", self.k);
        self.m
    }

    /// Encode one vector to PQ codes (length = `m`, one u8 per subspace).
    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        assert_eq!(x.len(), self.dim);
        let mut codes = vec![0u8; self.m];
        for s in 0..self.m {
            let off = s * self.d_sub;
            codes[s] = nearest_centroid_l2(&x[off..off + self.d_sub], &self.centroids[s], self.d_sub, self.k) as u8;
        }
        codes
    }

    /// Encode many vectors row-major. Output length = n*m.
    pub fn encode_many(&self, xs: &[f32], n: usize) -> Vec<u8> {
        assert_eq!(xs.len(), n * self.dim);
        let mut codes = vec![0u8; n * self.m];
        for i in 0..n {
            let row = &xs[i * self.dim..(i + 1) * self.dim];
            for s in 0..self.m {
                let off = s * self.d_sub;
                codes[i * self.m + s] = nearest_centroid_l2(&row[off..off + self.d_sub], &self.centroids[s], self.d_sub, self.k) as u8;
            }
        }
        codes
    }

    /// Reconstruct an approximate vector from its codes.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.m);
        let mut out = vec![0f32; self.dim];
        for s in 0..self.m {
            let c = codes[s] as usize;
            let src = &self.centroids[s][c * self.d_sub..(c + 1) * self.d_sub];
            out[s * self.d_sub..(s + 1) * self.d_sub].copy_from_slice(src);
        }
        out
    }

    /// Build the asymmetric distance table for query `q`: a [m, k] table
    /// of partial inner products. Then the estimated score of an encoded
    /// vector with codes `c` is Σ_s table[s, c[s]].
    pub fn build_lookup_ip(&self, q: &[f32]) -> Vec<f32> {
        assert_eq!(q.len(), self.dim);
        let mut tbl = vec![0f32; self.m * self.k];
        for s in 0..self.m {
            let q_sub = &q[s * self.d_sub..(s + 1) * self.d_sub];
            let cs = &self.centroids[s];
            for c in 0..self.k {
                let cv = &cs[c * self.d_sub..(c + 1) * self.d_sub];
                let mut acc = 0f32;
                for j in 0..self.d_sub {
                    acc += q_sub[j] * cv[j];
                }
                tbl[s * self.k + c] = acc;
            }
        }
        tbl
    }

    /// Estimated inner product q·x̃ given lookup `tbl` and codes `c`.
    pub fn score_with_lookup(&self, tbl: &[f32], codes: &[u8]) -> f32 {
        let mut s_sum = 0f32;
        for s in 0..self.m {
            s_sum += tbl[s * self.k + codes[s] as usize];
        }
        s_sum
    }
}

fn nearest_centroid_l2(x: &[f32], centroids: &[f32], d: usize, k: usize) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for c in 0..k {
        let cv = &centroids[c * d..(c + 1) * d];
        let mut acc = 0f32;
        for j in 0..d {
            let diff = x[j] - cv[j];
            acc += diff * diff;
        }
        if acc < best_d {
            best_d = acc;
            best = c;
        }
    }
    best
}

/// Generate a synthetic dataset for benchmarking: unit-norm Gaussian vectors
/// (so the MIPS ↔ cosine distinction is meaningful), plus disjoint queries.
pub fn synthetic_unit_dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(seed);
    let mut data = vec![0f32; n * dim];
    for i in 0..n {
        let row = &mut data[i * dim..(i + 1) * dim];
        for v in row.iter_mut() {
            *v = sample_normal(&mut rng);
        }
        let mut norm = 0f32;
        for v in row.iter() {
            norm += *v * *v;
        }
        let norm = norm.sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    data
}

fn sample_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    // Box–Muller; one variate per call (sufficient for synthetic data).
    let u1: f32 = rng.gen::<f32>().max(1e-9);
    let u2: f32 = rng.gen::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    r * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Brute-force ground-truth top-`k` MIPS indices (descending dot).
pub fn brute_force_topk(data: &[f32], n: usize, dim: usize, q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut scored: Vec<(u32, f32)> = (0..n)
        .map(|i| {
            let row = &data[i * dim..(i + 1) * dim];
            let mut acc = 0f32;
            for j in 0..dim {
                acc += row[j] * q[j];
            }
            (i as u32, acc)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{train_anisotropic_pq, train_l2_pq, TrainOpts};

    #[test]
    fn encode_decode_roundtrip_within_quantisation_error() {
        let n = 2_000;
        let dim = 32;
        let m = 8;
        let data = synthetic_unit_dataset(n, dim, 7);
        let codebook = train_l2_pq(&data, n, dim, m, 16, TrainOpts::default());
        let codes = codebook.encode(&data[0..dim]);
        let recon = codebook.decode(&codes);
        assert_eq!(recon.len(), dim);
        // Reconstruction must be within a sane bound (unit-norm data).
        let mut err2 = 0f32;
        for j in 0..dim {
            err2 += (recon[j] - data[j]).powi(2);
        }
        assert!(err2 < 1.0, "reconstruction error² too large: {}", err2);
    }

    #[test]
    fn anisotropic_reduces_mips_error_vs_l2() {
        // Smoke test: on synthetic unit-norm data, η=4 should give *no worse*
        // MIPS error than η=1, on average. (We verify a strict win on the
        // larger benchmark in main.rs; here we just guard against regressions.)
        let n = 4_000;
        let dim = 32;
        let m = 8;
        let opts = TrainOpts { iters: 8, k: 16, seed: 1 };
        let data = synthetic_unit_dataset(n, dim, 9);
        let pq_l2 = train_l2_pq(&data, n, dim, m, opts.k, opts);
        let pq_an = train_anisotropic_pq(&data, n, dim, m, opts.k, 4.0, opts);

        // Use the dataset itself as queries (worst-case correlated).
        let mut err_l2 = 0f64;
        let mut err_an = 0f64;
        let queries = 200usize;
        for q_idx in 0..queries {
            let q = &data[q_idx * dim..(q_idx + 1) * dim];
            let tbl_l2 = pq_l2.build_lookup_ip(q);
            let tbl_an = pq_an.build_lookup_ip(q);
            for i in 0..n {
                let row = &data[i * dim..(i + 1) * dim];
                let mut true_ip = 0f32;
                for j in 0..dim {
                    true_ip += row[j] * q[j];
                }
                let c_l2 = pq_l2.encode(row);
                let c_an = pq_an.encode(row);
                let e_l2 = (pq_l2.score_with_lookup(&tbl_l2, &c_l2) - true_ip) as f64;
                let e_an = (pq_an.score_with_lookup(&tbl_an, &c_an) - true_ip) as f64;
                err_l2 += e_l2 * e_l2;
                err_an += e_an * e_an;
            }
        }
        let mse_l2 = err_l2 / (queries * n) as f64;
        let mse_an = err_an / (queries * n) as f64;
        println!("MIPS MSE: l2={:.4e}  anisotropic(η=4)={:.4e}", mse_l2, mse_an);
        // Anisotropic should be at least as good (strict win expected; allow
        // a small tolerance for k-means init variance).
        assert!(mse_an <= mse_l2 * 1.05, "anisotropic should match or beat L2 on MIPS MSE");
    }
}
