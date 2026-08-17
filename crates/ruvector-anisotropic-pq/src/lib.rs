//! # ruvector-anisotropic-pq
//!
//! Anisotropic (score-aware) Product Quantization.
//!
//! Standard PQ minimizes reconstruction L2 error, which is a poor proxy for
//! inner-product/MIPS retrieval quality: errors along the query direction hurt
//! score preservation more than perpendicular errors of equal magnitude. This
//! crate implements a per-subvector anisotropic loss (Guo et al., ICML 2020,
//! "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"):
//!
//! ```text
//!   L_aniso(s, c; eta) = ||s - c||^2 + (eta - 1) * (( (s - c) . u )^2 )
//! ```
//!
//! where `u = s / ||s||` is the unit-normalized subvector direction and
//! `eta >= 1` is the anisotropic weight. `eta = 1` recovers plain L2 PQ.
//!
//! The centroid update under this loss has a closed-form d_sub x d_sub linear
//! solve derived from:
//!
//! ```text
//!   sum_i (I + (eta - 1) u_i u_i^T) (s_i - c) = 0
//!   =>  A c = b  with  A = sum_i (I + (eta - 1) u_i u_i^T),
//!                      b = sum_i (I + (eta - 1) u_i u_i^T) s_i
//! ```
//!
//! Solved by Gauss-Jordan (d_sub is small: 4..16 typically). See
//! [`train_codebook`] for the inner loop.
//!
//! ## Backend trait
//! Backends implement [`PqTrainer`]. Three ship in-tree:
//! - [`L2Trainer`]           — baseline unweighted Lloyd's k-means
//! - [`NormWeightedTrainer`] — weight each vector by `||x||^2` (score-aware lite)
//! - [`AnisotropicTrainer`]  — full closed-form anisotropic loss

#![forbid(unsafe_code)]

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use thiserror::Error;

/// PQ error surface.
#[derive(Debug, Error)]
pub enum PqError {
    #[error("dimension {dim} not divisible by {m} subquantizers")]
    BadDim { dim: usize, m: usize },
    #[error("k ({k}) must be <= number of training vectors ({n})")]
    TooFewTrainingVectors { k: usize, n: usize },
    #[error("empty training set")]
    Empty,
    #[error("singular normal equations in anisotropic centroid update")]
    Singular,
}

/// Product-quantization codebook: `m` subquantizers, each with `k` centroids
/// of dimension `d_sub = dim / m`.
#[derive(Clone, Debug)]
pub struct PqCodebook {
    pub dim: usize,
    pub m: usize,
    pub k: usize,
    pub d_sub: usize,
    /// Flat layout: `centroids[sub * k * d_sub + code * d_sub + j]`.
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Return the k centroids for subquantizer `sub`, as a flat slice of length `k * d_sub`.
    #[inline]
    pub fn sub_centroids(&self, sub: usize) -> &[f32] {
        let off = sub * self.k * self.d_sub;
        &self.centroids[off..off + self.k * self.d_sub]
    }

    /// Encode `x` (length `dim`) into `m` codes (u8, so `k <= 256`).
    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        assert_eq!(x.len(), self.dim);
        assert!(self.k <= 256, "encode() supports k<=256 (u8 codes)");
        let mut codes = vec![0u8; self.m];
        for sub in 0..self.m {
            let x_sub = &x[sub * self.d_sub..(sub + 1) * self.d_sub];
            let cents = self.sub_centroids(sub);
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let cent = &cents[c * self.d_sub..(c + 1) * self.d_sub];
                let d = sqeuclid(x_sub, cent);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            codes[sub] = best as u8;
        }
        codes
    }

    /// Reconstruct a vector from its PQ codes.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        assert_eq!(codes.len(), self.m);
        let mut out = vec![0f32; self.dim];
        for sub in 0..self.m {
            let cents = self.sub_centroids(sub);
            let c = codes[sub] as usize;
            let cent = &cents[c * self.d_sub..(c + 1) * self.d_sub];
            out[sub * self.d_sub..(sub + 1) * self.d_sub].copy_from_slice(cent);
        }
        out
    }

    /// Bytes per encoded vector (assumes k <= 256).
    pub fn bytes_per_vector(&self) -> usize {
        self.m
    }
}

/// Trainer backend trait. Implement to plug in new PQ variants.
pub trait PqTrainer {
    /// Human-readable variant name (used in benchmark reports).
    fn name(&self) -> &'static str;
    /// Train a codebook from `data` (row-major, `n * dim`).
    fn train(&self, data: &[f32], n: usize, dim: usize, m: usize, k: usize) -> Result<PqCodebook, PqError>;
}

/// Standard L2-loss PQ: unweighted Lloyd's k-means per subquantizer.
pub struct L2Trainer {
    pub iters: usize,
    pub seed: u64,
}
impl Default for L2Trainer {
    fn default() -> Self {
        Self { iters: 25, seed: 42 }
    }
}
impl PqTrainer for L2Trainer {
    fn name(&self) -> &'static str {
        "L2-PQ"
    }
    fn train(&self, data: &[f32], n: usize, dim: usize, m: usize, k: usize) -> Result<PqCodebook, PqError> {
        train_codebook(data, n, dim, m, k, self.iters, self.seed, LossKind::L2)
    }
}

/// Score-aware "lite": weight each vector by `||x||^2` in centroid updates.
/// Higher-norm vectors dominate MIPS ranking, so quantizing them more
/// accurately is a cheap approximation of the anisotropic loss.
pub struct NormWeightedTrainer {
    pub iters: usize,
    pub seed: u64,
}
impl Default for NormWeightedTrainer {
    fn default() -> Self {
        Self { iters: 25, seed: 42 }
    }
}
impl PqTrainer for NormWeightedTrainer {
    fn name(&self) -> &'static str {
        "NormWeighted-PQ"
    }
    fn train(&self, data: &[f32], n: usize, dim: usize, m: usize, k: usize) -> Result<PqCodebook, PqError> {
        train_codebook(data, n, dim, m, k, self.iters, self.seed, LossKind::NormWeighted)
    }
}

/// Full anisotropic loss with closed-form centroid update. `eta >= 1`:
/// `eta = 1` recovers L2; typical MIPS sweet spot is eta in \[2, 8\].
pub struct AnisotropicTrainer {
    pub iters: usize,
    pub seed: u64,
    pub eta: f32,
}
impl AnisotropicTrainer {
    pub fn new(eta: f32) -> Self {
        Self { iters: 25, seed: 42, eta }
    }
}
impl PqTrainer for AnisotropicTrainer {
    fn name(&self) -> &'static str {
        "Anisotropic-PQ"
    }
    fn train(&self, data: &[f32], n: usize, dim: usize, m: usize, k: usize) -> Result<PqCodebook, PqError> {
        train_codebook(data, n, dim, m, k, self.iters, self.seed, LossKind::Anisotropic(self.eta))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LossKind {
    L2,
    NormWeighted,
    Anisotropic(f32),
}

/// Core training loop. Shared across variants for a fair benchmark
/// (only the assignment/update math differs).
pub fn train_codebook(
    data: &[f32],
    n: usize,
    dim: usize,
    m: usize,
    k: usize,
    iters: usize,
    seed: u64,
    loss: LossKind,
) -> Result<PqCodebook, PqError> {
    if n == 0 {
        return Err(PqError::Empty);
    }
    if dim % m != 0 {
        return Err(PqError::BadDim { dim, m });
    }
    if k > n {
        return Err(PqError::TooFewTrainingVectors { k, n });
    }
    let d_sub = dim / m;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = vec![0f32; m * k * d_sub];

    // Pre-compute per-full-vector norm (for NormWeighted / Anisotropic).
    let full_norm2: Vec<f32> = (0..n)
        .map(|i| {
            let x = &data[i * dim..(i + 1) * dim];
            x.iter().map(|&v| v * v).sum::<f32>().max(1e-12)
        })
        .collect();

    for sub in 0..m {
        // seed by sampling k distinct training vectors' subvectors
        let mut idx: Vec<usize> = (0..n).collect();
        idx.shuffle(&mut rng);
        for c in 0..k {
            let src = &data[idx[c] * dim + sub * d_sub..idx[c] * dim + (sub + 1) * d_sub];
            let dst_off = sub * k * d_sub + c * d_sub;
            centroids[dst_off..dst_off + d_sub].copy_from_slice(src);
        }

        // Lloyd's iterations
        let mut assign = vec![0u16; n];
        for _it in 0..iters {
            // ---- assignment ----
            for i in 0..n {
                let x_sub = &data[i * dim + sub * d_sub..i * dim + (sub + 1) * d_sub];
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let cent = &centroids[sub * k * d_sub + c * d_sub..sub * k * d_sub + (c + 1) * d_sub];
                    let d = match loss {
                        LossKind::L2 | LossKind::NormWeighted => sqeuclid(x_sub, cent),
                        LossKind::Anisotropic(eta) => aniso_dist(x_sub, cent, eta),
                    };
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                assign[i] = best as u16;
            }

            // ---- update ----
            match loss {
                LossKind::L2 => {
                    update_l2(data, dim, sub, d_sub, k, n, &assign, &mut centroids, &mut rng);
                }
                LossKind::NormWeighted => {
                    update_norm_weighted(
                        data, dim, sub, d_sub, k, n, &assign, &full_norm2, &mut centroids, &mut rng,
                    );
                }
                LossKind::Anisotropic(eta) => {
                    update_anisotropic(
                        data, dim, sub, d_sub, k, n, &assign, eta, &mut centroids, &mut rng,
                    )?;
                }
            }
        }
    }

    Ok(PqCodebook { dim, m, k, d_sub, centroids })
}

// ---- distance / loss primitives -------------------------------------------

#[inline]
fn sqeuclid(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Anisotropic per-subvector distance with unit direction u = s / ||s||:
/// `||s - c||^2 + (eta - 1) * (((s - c) . u))^2`.
#[inline]
fn aniso_dist(s: &[f32], c: &[f32], eta: f32) -> f32 {
    let sn2 = dot(s, s).max(1e-12);
    let sn = sn2.sqrt();
    let base = sqeuclid(s, c);
    let mut proj = 0f32;
    for i in 0..s.len() {
        proj += (s[i] - c[i]) * (s[i] / sn);
    }
    base + (eta - 1.0) * proj * proj
}

// ---- centroid updates -----------------------------------------------------

fn update_l2(
    data: &[f32], dim: usize, sub: usize, d_sub: usize, k: usize, n: usize,
    assign: &[u16], centroids: &mut [f32], rng: &mut StdRng,
) {
    let mut sums = vec![0f32; k * d_sub];
    let mut cnts = vec![0u32; k];
    for i in 0..n {
        let c = assign[i] as usize;
        cnts[c] += 1;
        let x_sub = &data[i * dim + sub * d_sub..i * dim + (sub + 1) * d_sub];
        let dst = &mut sums[c * d_sub..(c + 1) * d_sub];
        for j in 0..d_sub {
            dst[j] += x_sub[j];
        }
    }
    for c in 0..k {
        let dst = &mut centroids[sub * k * d_sub + c * d_sub..sub * k * d_sub + (c + 1) * d_sub];
        if cnts[c] == 0 {
            let i = rng.gen_range(0..n);
            let src = &data[i * dim + sub * d_sub..i * dim + (sub + 1) * d_sub];
            dst.copy_from_slice(src);
            continue;
        }
        let inv = 1.0 / (cnts[c] as f32);
        for j in 0..d_sub {
            dst[j] = sums[c * d_sub + j] * inv;
        }
    }
}

fn update_norm_weighted(
    data: &[f32], dim: usize, sub: usize, d_sub: usize, k: usize, n: usize,
    assign: &[u16], full_norm2: &[f32], centroids: &mut [f32], rng: &mut StdRng,
) {
    let mut sums = vec![0f32; k * d_sub];
    let mut wsum = vec![0f32; k];
    for i in 0..n {
        let c = assign[i] as usize;
        let w = full_norm2[i];
        wsum[c] += w;
        let x_sub = &data[i * dim + sub * d_sub..i * dim + (sub + 1) * d_sub];
        let dst = &mut sums[c * d_sub..(c + 1) * d_sub];
        for j in 0..d_sub {
            dst[j] += x_sub[j] * w;
        }
    }
    for c in 0..k {
        let dst = &mut centroids[sub * k * d_sub + c * d_sub..sub * k * d_sub + (c + 1) * d_sub];
        if wsum[c] < 1e-12 {
            let i = rng.gen_range(0..n);
            let src = &data[i * dim + sub * d_sub..i * dim + (sub + 1) * d_sub];
            dst.copy_from_slice(src);
            continue;
        }
        let inv = 1.0 / wsum[c];
        for j in 0..d_sub {
            dst[j] = sums[c * d_sub + j] * inv;
        }
    }
}

/// Closed-form anisotropic update: solve `A c = b` per cluster where
/// `A = sum (I + (eta-1) u u^T)` and `b = sum (I + (eta-1) u u^T) s`,
/// with `u = s / ||s||` (subvector unit direction).
fn update_anisotropic(
    data: &[f32], dim: usize, sub: usize, d_sub: usize, k: usize, n: usize,
    assign: &[u16], eta: f32, centroids: &mut [f32], rng: &mut StdRng,
) -> Result<(), PqError> {
    // Per-cluster running d_sub x d_sub matrix A and d_sub vector b.
    let dd = d_sub * d_sub;
    let mut a_mat = vec![0f32; k * dd];
    let mut b_vec = vec![0f32; k * d_sub];
    let mut cnts = vec![0u32; k];

    for i in 0..n {
        let c = assign[i] as usize;
        cnts[c] += 1;
        let x_sub = &data[i * dim + sub * d_sub..i * dim + (sub + 1) * d_sub];
        let n2 = dot(x_sub, x_sub).max(1e-12);
        let inv_n = 1.0 / n2.sqrt();
        let a_off = c * dd;
        let b_off = c * d_sub;
        // Accumulate I contribution
        for j in 0..d_sub {
            a_mat[a_off + j * d_sub + j] += 1.0;
            b_vec[b_off + j] += x_sub[j];
        }
        // Accumulate (eta-1) * u u^T contribution and (eta-1) * u u^T s = (eta-1) * u * (u.s)
        // With u = x_sub * inv_n, and u.s = ||x_sub||, so u * (u.s) = x_sub * inv_n * ||x_sub|| = x_sub
        // => rhs adds (eta - 1) * x_sub_j
        let em1 = eta - 1.0;
        for jr in 0..d_sub {
            let ur = x_sub[jr] * inv_n;
            b_vec[b_off + jr] += em1 * x_sub[jr];
            for jc in 0..d_sub {
                let uc = x_sub[jc] * inv_n;
                a_mat[a_off + jr * d_sub + jc] += em1 * ur * uc;
            }
        }
    }

    // Solve per cluster
    for c in 0..k {
        let dst = &mut centroids[sub * k * d_sub + c * d_sub..sub * k * d_sub + (c + 1) * d_sub];
        if cnts[c] == 0 {
            let i = rng.gen_range(0..n);
            let src = &data[i * dim + sub * d_sub..i * dim + (sub + 1) * d_sub];
            dst.copy_from_slice(src);
            continue;
        }
        let a = &mut a_mat[c * dd..(c + 1) * dd];
        let b = &mut b_vec[c * d_sub..(c + 1) * d_sub];
        gauss_solve(a, b, d_sub)?;
        dst.copy_from_slice(b);
    }
    Ok(())
}

/// In-place Gauss-Jordan solve. `a` is `d x d` row-major, `b` is `d`.
/// On success `b` holds the solution `x = A^-1 b`.
fn gauss_solve(a: &mut [f32], b: &mut [f32], d: usize) -> Result<(), PqError> {
    for i in 0..d {
        // partial pivot
        let mut piv = i;
        let mut pv = a[i * d + i].abs();
        for r in (i + 1)..d {
            let v = a[r * d + i].abs();
            if v > pv {
                pv = v;
                piv = r;
            }
        }
        if pv < 1e-10 {
            return Err(PqError::Singular);
        }
        if piv != i {
            for j in 0..d {
                a.swap(i * d + j, piv * d + j);
            }
            b.swap(i, piv);
        }
        let inv = 1.0 / a[i * d + i];
        for j in 0..d {
            a[i * d + j] *= inv;
        }
        b[i] *= inv;
        for r in 0..d {
            if r == i {
                continue;
            }
            let f = a[r * d + i];
            if f.abs() < 1e-20 {
                continue;
            }
            for j in 0..d {
                a[r * d + j] -= f * a[i * d + j];
            }
            b[r] -= f * b[i];
        }
    }
    Ok(())
}

/// Compute recall@k: fraction of true top-k ground-truth neighbors that appear
/// in the returned top-k list (ties are broken by index).
pub fn recall_at_k(returned: &[u32], truth: &[u32], k: usize) -> f32 {
    use std::collections::HashSet;
    let truth_set: HashSet<u32> = truth.iter().take(k).copied().collect();
    let hit = returned.iter().take(k).filter(|x| truth_set.contains(x)).count();
    hit as f32 / k as f32
}

/// Compute exact inner-product top-k for benchmark ground truth.
pub fn exact_ip_topk(base: &[f32], n: usize, dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, dot(&base[i * dim..(i + 1) * dim], query)))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

/// PQ-approximate inner-product top-k using asymmetric distance tables
/// (`ADC`): query is scored against reconstructed database via subvector
/// dot-product lookup tables.
pub fn pq_ip_topk(cb: &PqCodebook, codes: &[u8], n: usize, query: &[f32], k: usize) -> Vec<u32> {
    // Build per-subquantizer lookup: dot(query_sub, centroid_c)
    let mut lut = vec![0f32; cb.m * cb.k];
    for sub in 0..cb.m {
        let q_sub = &query[sub * cb.d_sub..(sub + 1) * cb.d_sub];
        let cents = cb.sub_centroids(sub);
        for c in 0..cb.k {
            let cent = &cents[c * cb.d_sub..(c + 1) * cb.d_sub];
            lut[sub * cb.k + c] = dot(q_sub, cent);
        }
    }
    let mut scored: Vec<(u32, f32)> = (0..n)
        .map(|i| {
            let mut s = 0f32;
            for sub in 0..cb.m {
                let code = codes[i * cb.m + sub] as usize;
                s += lut[sub * cb.k + code];
            }
            (i as u32, s)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand_chacha::ChaCha8Rng;

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = <ChaCha8Rng as SeedableRng>::seed_from_u64(seed);
        (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }

    #[test]
    fn gauss_identity_returns_rhs() {
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = vec![3.0, -1.5, 7.25];
        gauss_solve(&mut a, &mut b, 3).unwrap();
        assert!((b[0] - 3.0).abs() < 1e-6);
        assert!((b[1] + 1.5).abs() < 1e-6);
        assert!((b[2] - 7.25).abs() < 1e-6);
    }

    #[test]
    fn gauss_solves_2x2() {
        // [[2,1],[1,3]] x = [5,10] -> x = [1,3]
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 10.0];
        gauss_solve(&mut a, &mut b, 2).unwrap();
        assert!((b[0] - 1.0).abs() < 1e-5);
        assert!((b[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn l2_train_produces_valid_codebook() {
        let n = 200; let dim = 16; let m = 4; let k = 8;
        let data = synth(n, dim, 1);
        let cb = L2Trainer::default().train(&data, n, dim, m, k).unwrap();
        assert_eq!(cb.centroids.len(), m * k * (dim / m));
        let codes = cb.encode(&data[0..dim]);
        assert_eq!(codes.len(), m);
        let rec = cb.decode(&codes);
        assert_eq!(rec.len(), dim);
    }

    #[test]
    fn anisotropic_matches_l2_at_eta_one() {
        // eta=1.0: anisotropic loss reduces to L2, so codebooks should be
        // numerically identical up to seed/order effects.
        let n = 150; let dim = 8; let m = 2; let k = 4;
        let data = synth(n, dim, 42);
        let l2 = L2Trainer { iters: 15, seed: 7 }.train(&data, n, dim, m, k).unwrap();
        let an = AnisotropicTrainer { iters: 15, seed: 7, eta: 1.0 }
            .train(&data, n, dim, m, k).unwrap();
        // Sum of nearest-code reconstruction MSE should match to within 1%
        let mse = |cb: &PqCodebook| -> f32 {
            let mut s = 0f32;
            for i in 0..n {
                let x = &data[i * dim..(i + 1) * dim];
                let r = cb.decode(&cb.encode(x));
                for j in 0..dim {
                    let d = x[j] - r[j];
                    s += d * d;
                }
            }
            s / n as f32
        };
        let a = mse(&l2); let b = mse(&an);
        assert!((a - b).abs() / a < 0.02, "L2 mse={} aniso(eta=1) mse={}", a, b);
    }

    #[test]
    fn norm_weighted_beats_l2_on_norm_biased_data() {
        // Practical finding: score-aware norm-weighting improves MIPS recall
        // over vanilla L2-PQ on heavy-tail-norm data (which mimics real
        // recommender embeddings). See docs/research/nightly for full writeup.
        let n = 400; let dim = 16; let m = 4; let k = 16;
        let mut data = synth(n, dim, 3);
        for i in 0..100 {
            for j in 0..dim {
                data[i * dim + j] *= 3.0;
            }
        }
        let l2 = L2Trainer { iters: 20, seed: 11 }.train(&data, n, dim, m, k).unwrap();
        let nw = NormWeightedTrainer { iters: 20, seed: 11 }.train(&data, n, dim, m, k).unwrap();

        let codes_l2: Vec<u8> = (0..n).flat_map(|i| l2.encode(&data[i * dim..(i + 1) * dim])).collect();
        let codes_nw: Vec<u8> = (0..n).flat_map(|i| nw.encode(&data[i * dim..(i + 1) * dim])).collect();

        let queries = synth(20, dim, 99);
        let mut r_l2 = 0f32; let mut r_nw = 0f32;
        let topk = 10;
        for q in 0..20 {
            let q_vec = &queries[q * dim..(q + 1) * dim];
            let truth = exact_ip_topk(&data, n, dim, q_vec, topk);
            r_l2 += recall_at_k(&pq_ip_topk(&l2, &codes_l2, n, q_vec, topk), &truth, topk);
            r_nw += recall_at_k(&pq_ip_topk(&nw, &codes_nw, n, q_vec, topk), &truth, topk);
        }
        r_l2 /= 20.0; r_nw /= 20.0;
        assert!(r_nw >= r_l2, "norm-weighted recall {} should be >= L2 recall {} on norm-biased data", r_nw, r_l2);
    }
}
