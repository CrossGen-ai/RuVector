//! Anisotropic Product Quantization (score-aware quantization, ScaNN-style).
//!
//! Reference: Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar.
//! "Accelerating Large-Scale Inference with Anisotropic Vector Quantization."
//! ICML 2020. https://arxiv.org/abs/1908.10396
//!
//! This crate implements two variants behind a common `Quantizer` trait:
//!
//! - [`Pq`]    — standard Product Quantization (isotropic ℓ² loss, Lloyd's).
//! - [`ApqQuantizer`] — Anisotropic PQ with a per-subspace score-aware loss
//!   that up-weights residual error parallel to the datapoint's direction.
//!
//! Per-subspace loss (η ≥ 1):
//!     L(r, x_sub) = η · (r · x̂_sub)² + ‖r_⊥‖²
//! where r = x_sub − c, x̂_sub = x_sub / ‖x_sub‖, r_∥ = (r · x̂_sub) x̂_sub.
//!
//! η = 1 recovers standard PQ. For η > 1, centroid updates solve a small
//! per-centroid linear system with weight matrix W_i = I + (η−1) x̂ x̂ᵀ.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApqError {
    #[error("dimension {dim} is not divisible by subspace count {m}")]
    DimMismatch { dim: usize, m: usize },
    #[error("empty training set")]
    EmptyTrainingSet,
    #[error("k ({k}) exceeds training-set size ({n}) for a subspace")]
    InsufficientData { k: usize, n: usize },
}

/// Common interface for both PQ and Anisotropic PQ.
pub trait Quantizer: Sync {
    fn dim(&self) -> usize;
    fn m(&self) -> usize;
    fn k(&self) -> usize;

    /// Encode a single vector into M codes in [0, K).
    fn encode(&self, x: &[f32]) -> Vec<u8>;

    /// Encode a batch of vectors.
    fn encode_batch(&self, xs: &[Vec<f32>]) -> Vec<Vec<u8>> {
        xs.iter().map(|x| self.encode(x)).collect()
    }

    /// Build a query-side lookup table: LUT[m * K + k] = q^(m) · c_{m,k}.
    /// Used for asymmetric maximum-inner-product scoring.
    fn build_ip_lut(&self, q: &[f32]) -> Vec<f32>;

    /// Score an encoded vector against the LUT (inner product).
    fn score(&self, lut: &[f32], code: &[u8]) -> f32 {
        let k = self.k();
        let mut s = 0.0_f32;
        for (m, &c) in code.iter().enumerate() {
            s += lut[m * k + c as usize];
        }
        s
    }

    /// Estimated bytes-per-vector after encoding (codes only, M * log2(K) bits).
    fn bytes_per_vector(&self) -> usize {
        let bits = self.m() * (self.k() as f64).log2().ceil() as usize;
        (bits + 7) / 8
    }
}

/// Standard Product Quantization.
#[derive(Clone)]
pub struct Pq {
    dim: usize,
    m: usize,
    k: usize,
    sub_dim: usize,
    /// codebooks[m]: K × sub_dim, row-major.
    codebooks: Vec<Vec<f32>>,
}

impl Pq {
    pub fn train(
        data: &[Vec<f32>],
        m: usize,
        k: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self, ApqError> {
        if data.is_empty() {
            return Err(ApqError::EmptyTrainingSet);
        }
        let dim = data[0].len();
        if dim % m != 0 {
            return Err(ApqError::DimMismatch { dim, m });
        }
        if k > data.len() {
            return Err(ApqError::InsufficientData { k, n: data.len() });
        }
        let sub_dim = dim / m;
        let mut codebooks = Vec::with_capacity(m);
        for sub in 0..m {
            let subset: Vec<Vec<f32>> = data
                .iter()
                .map(|v| v[sub * sub_dim..(sub + 1) * sub_dim].to_vec())
                .collect();
            let cb = kmeans_isotropic(&subset, k, sub_dim, iters, seed + sub as u64);
            codebooks.push(flatten(&cb));
        }
        Ok(Self { dim, m, k, sub_dim, codebooks })
    }
}

impl Quantizer for Pq {
    fn dim(&self) -> usize { self.dim }
    fn m(&self) -> usize { self.m }
    fn k(&self) -> usize { self.k }

    fn encode(&self, x: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let off = sub * self.sub_dim;
            let xs = &x[off..off + self.sub_dim];
            let cb = &self.codebooks[sub];
            let mut best = 0_usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let co = c * self.sub_dim;
                let d = sqdist(xs, &cb[co..co + self.sub_dim]);
                if d < best_d { best_d = d; best = c; }
            }
            codes.push(best as u8);
        }
        codes
    }

    fn build_ip_lut(&self, q: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0_f32; self.m * self.k];
        for sub in 0..self.m {
            let off = sub * self.sub_dim;
            let qs = &q[off..off + self.sub_dim];
            let cb = &self.codebooks[sub];
            for c in 0..self.k {
                let co = c * self.sub_dim;
                lut[sub * self.k + c] = dot(qs, &cb[co..co + self.sub_dim]);
            }
        }
        lut
    }
}

/// Anisotropic Product Quantization (score-aware loss).
#[derive(Clone)]
pub struct ApqQuantizer {
    dim: usize,
    m: usize,
    k: usize,
    sub_dim: usize,
    eta: f32,
    codebooks: Vec<Vec<f32>>,
}

impl ApqQuantizer {
    /// `eta = 1.0` reproduces standard PQ; eta > 1 amplifies parallel-error
    /// penalty. Typical values: 2.0 – 4.0.
    pub fn train(
        data: &[Vec<f32>],
        m: usize,
        k: usize,
        eta: f32,
        iters: usize,
        seed: u64,
    ) -> Result<Self, ApqError> {
        if data.is_empty() {
            return Err(ApqError::EmptyTrainingSet);
        }
        let dim = data[0].len();
        if dim % m != 0 {
            return Err(ApqError::DimMismatch { dim, m });
        }
        if k > data.len() {
            return Err(ApqError::InsufficientData { k, n: data.len() });
        }
        let sub_dim = dim / m;
        let mut codebooks = Vec::with_capacity(m);
        for sub in 0..m {
            let mut subset = Vec::with_capacity(data.len());
            let mut dirs = Vec::with_capacity(data.len());
            for v in data {
                let xs = v[sub * sub_dim..(sub + 1) * sub_dim].to_vec();
                let n = norm(&xs).max(1e-12);
                let dir: Vec<f32> = xs.iter().map(|&a| a / n).collect();
                subset.push(xs);
                dirs.push(dir);
            }
            let cb = kmeans_anisotropic(&subset, &dirs, k, sub_dim, eta, iters, seed + sub as u64);
            codebooks.push(flatten(&cb));
        }
        Ok(Self { dim, m, k, sub_dim, eta, codebooks })
    }

    pub fn eta(&self) -> f32 { self.eta }
}

impl Quantizer for ApqQuantizer {
    fn dim(&self) -> usize { self.dim }
    fn m(&self) -> usize { self.m }
    fn k(&self) -> usize { self.k }

    fn encode(&self, x: &[f32]) -> Vec<u8> {
        let eta = self.eta;
        let mut codes = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let off = sub * self.sub_dim;
            let xs = &x[off..off + self.sub_dim];
            let n = norm(xs).max(1e-12);
            let inv_n = 1.0 / n;
            let cb = &self.codebooks[sub];
            let mut best = 0_usize;
            let mut best_l = f32::INFINITY;
            for c in 0..self.k {
                let co = c * self.sub_dim;
                let cc = &cb[co..co + self.sub_dim];
                // residual r = xs - cc
                let mut par = 0.0_f32; // r · x̂
                let mut sq = 0.0_f32;  // ||r||²
                for i in 0..self.sub_dim {
                    let r = xs[i] - cc[i];
                    par += r * xs[i] * inv_n;
                    sq  += r * r;
                }
                let perp = sq - par * par;
                let loss = eta * par * par + perp;
                if loss < best_l { best_l = loss; best = c; }
            }
            codes.push(best as u8);
        }
        codes
    }

    fn build_ip_lut(&self, q: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0_f32; self.m * self.k];
        for sub in 0..self.m {
            let off = sub * self.sub_dim;
            let qs = &q[off..off + self.sub_dim];
            let cb = &self.codebooks[sub];
            for c in 0..self.k {
                let co = c * self.sub_dim;
                lut[sub * self.k + c] = dot(qs, &cb[co..co + self.sub_dim]);
            }
        }
        lut
    }
}

// --- helpers ---------------------------------------------------------------

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

fn sqdist(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

fn norm(a: &[f32]) -> f32 { dot(a, a).sqrt() }

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * rows[0].len());
    for r in rows { out.extend_from_slice(r); }
    out
}

fn kmeans_isotropic(
    data: &[Vec<f32>], k: usize, d: usize, iters: usize, seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vec<f32>> = data
        .choose_multiple(&mut rng, k)
        .cloned()
        .collect();
    for _ in 0..iters {
        let mut sums = vec![vec![0.0_f32; d]; k];
        let mut cnt = vec![0_usize; k];
        for x in data {
            let mut best = 0;
            let mut best_d = f32::INFINITY;
            for (c, cv) in centroids.iter().enumerate() {
                let dd = sqdist(x, cv);
                if dd < best_d { best_d = dd; best = c; }
            }
            for i in 0..d { sums[best][i] += x[i]; }
            cnt[best] += 1;
        }
        for c in 0..k {
            if cnt[c] > 0 {
                let inv = 1.0 / cnt[c] as f32;
                for i in 0..d { centroids[c][i] = sums[c][i] * inv; }
            } else {
                // reseed empty cluster
                let r = rng.gen_range(0..data.len());
                centroids[c] = data[r].clone();
            }
        }
    }
    centroids
}

/// k-means with anisotropic per-subspace loss.
/// Each point i contributes W_i = I + (η-1) n̂_i n̂_iᵀ.
fn kmeans_anisotropic(
    data: &[Vec<f32>], dirs: &[Vec<f32>], k: usize, d: usize,
    eta: f32, iters: usize, seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vec<f32>> = data
        .choose_multiple(&mut rng, k)
        .cloned()
        .collect();
    let alpha = eta - 1.0; // weight on parallel direction beyond identity
    for _ in 0..iters {
        // For each centroid c, accumulate A_c = Σ W_i, b_c = Σ W_i x_i.
        // W_i x_i = x_i + α (n̂_i · x_i) n̂_i.
        // Σ W_i = n_c · I + α Σ n̂_i n̂_iᵀ.
        let mut counts = vec![0_usize; k];
        let mut nnt = vec![vec![0.0_f32; d * d]; k]; // Σ n̂ n̂ᵀ
        let mut bvec = vec![vec![0.0_f32; d]; k];    // Σ W_i x_i

        for (x, n) in data.iter().zip(dirs.iter()) {
            let mut best = 0;
            let mut best_l = f32::INFINITY;
            for (c, cv) in centroids.iter().enumerate() {
                // loss = η * par² + perp = par² (η-1) + ||r||²
                let mut par = 0.0_f32;
                let mut sq = 0.0_f32;
                for i in 0..d {
                    let r = x[i] - cv[i];
                    par += r * n[i];
                    sq  += r * r;
                }
                let loss = sq + alpha * par * par;
                if loss < best_l { best_l = loss; best = c; }
            }
            counts[best] += 1;
            // accumulate b: x + α (n·x) n
            let nx = dot(n, x);
            for i in 0..d {
                bvec[best][i] += x[i] + alpha * nx * n[i];
            }
            // accumulate outer product n nᵀ
            let acc = &mut nnt[best];
            for i in 0..d {
                let ni = n[i];
                for j in 0..d {
                    acc[i * d + j] += ni * n[j];
                }
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                let r = rng.gen_range(0..data.len());
                centroids[c] = data[r].clone();
                continue;
            }
            // A = n_c I + α Σ n̂ n̂ᵀ
            let nc = counts[c] as f32;
            let mut a = vec![0.0_f32; d * d];
            for i in 0..d {
                for j in 0..d {
                    a[i * d + j] = alpha * nnt[c][i * d + j];
                }
                a[i * d + i] += nc;
            }
            let new = solve_linear(&mut a, &mut bvec[c].clone(), d);
            centroids[c] = new;
        }
    }
    centroids
}

/// Solve A x = b in place via Gauss-Jordan with partial pivoting.
/// A is d×d row-major; b is length d; returns x.
fn solve_linear(a: &mut [f32], b: &mut [f32], d: usize) -> Vec<f32> {
    for col in 0..d {
        // partial pivot
        let mut piv = col;
        let mut pv = a[col * d + col].abs();
        for r in (col + 1)..d {
            let v = a[r * d + col].abs();
            if v > pv { pv = v; piv = r; }
        }
        if pv < 1e-12 {
            // singular: fall back to identity for this row
            return b.to_vec();
        }
        if piv != col {
            for j in 0..d {
                a.swap(col * d + j, piv * d + j);
            }
            b.swap(col, piv);
        }
        let inv = 1.0 / a[col * d + col];
        for j in 0..d { a[col * d + j] *= inv; }
        b[col] *= inv;
        for r in 0..d {
            if r == col { continue; }
            let f = a[r * d + col];
            if f.abs() < 1e-20 { continue; }
            for j in 0..d {
                a[r * d + j] -= f * a[col * d + j];
            }
            b[r] -= f * b[col];
        }
    }
    b.to_vec()
}

// --- recall harness --------------------------------------------------------

/// Brute-force top-k inner-product neighbors (ground truth).
pub fn brute_force_topk(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, x)| (i, dot(query, x)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Approximate top-k inner-product neighbors via a quantizer's codes.
pub fn approx_topk<Q: Quantizer>(
    q: &Q, query: &[f32], codes: &[Vec<u8>], k: usize,
) -> Vec<usize> {
    let lut = q.build_ip_lut(query);
    let mut scored: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| (i, q.score(&lut, c)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Recall@k of `approx` against `truth`.
pub fn recall_at_k(approx: &[usize], truth: &[usize]) -> f32 {
    if truth.is_empty() { return 0.0; }
    let mut hits = 0;
    for t in truth {
        if approx.contains(t) { hits += 1; }
    }
    hits as f32 / truth.len() as f32
}
