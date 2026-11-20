//! Iterative Quantization (ITQ) for binary embedding codes.
//!
//! Gong & Lazebnik, "Iterative Quantization: A Procrustean Approach
//! to Learning Binary Codes", CVPR 2011. Given PCA-projected data V ∈ R^{n x b},
//! find an orthogonal rotation R ∈ R^{b x b} minimising the quantization error
//!   || B - V R ||_F^2     where B = sign(V R) ∈ {-1,+1}^{n x b}.
//! Alternating minimisation:
//!   (i) fix R → B = sign(V R)
//!  (ii) fix B → R = U Q^T where SVD(V^T B) = Q Σ U^T  (Procrustes).
//!
//! This crate ships:
//!  * [`ItqEncoder`] — fit PCA + ITQ rotation, encode vectors to packed bits
//!  * [`BinaryIndex`] — flat Hamming-distance brute-force index for codes
//!  * three swappable [`Rotation`] backends (Identity / Random / Itq) so
//!    improvements can be measured in isolation.
//!
//! No dependencies: linear algebra kernels (centred PCA via orthogonal
//! iteration, Jacobi SVD on the b×b Procrustes problem, popcount Hamming)
//! are implemented in-crate.

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]

use core::cmp::Reverse;
use core::fmt;

// ---------- rotation backend trait --------------------------------------------------

pub trait Rotation: fmt::Debug {
    fn name(&self) -> &'static str;
    /// Rotate a single projected vector in place (v ← v·R).
    fn apply(&self, v: &mut [f32]);
}

#[derive(Debug, Clone, Copy)]
pub struct IdentityRotation;
impl Rotation for IdentityRotation {
    fn name(&self) -> &'static str { "identity" }
    fn apply(&self, _v: &mut [f32]) {}
}

#[derive(Debug, Clone)]
pub struct DenseRotation {
    name: &'static str,
    bits: usize,
    /// row-major b×b orthogonal matrix.
    r: Vec<f32>,
}

impl DenseRotation {
    pub fn new(name: &'static str, bits: usize, r: Vec<f32>) -> Self {
        assert_eq!(r.len(), bits * bits);
        Self { name, bits, r }
    }
    pub fn matrix(&self) -> &[f32] { &self.r }
}

impl Rotation for DenseRotation {
    fn name(&self) -> &'static str { self.name }
    fn apply(&self, v: &mut [f32]) {
        assert_eq!(v.len(), self.bits);
        let mut tmp = vec![0f32; self.bits];
        for j in 0..self.bits {
            let mut s = 0.0f32;
            for i in 0..self.bits {
                s += v[i] * self.r[i * self.bits + j];
            }
            tmp[j] = s;
        }
        v.copy_from_slice(&tmp);
    }
}

// ---------- encoder: PCA + rotation + sign packing ----------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationKind { Identity, Random, Itq }

#[derive(Debug)]
pub struct ItqEncoder {
    pub dim: usize,
    pub bits: usize,
    pub mean: Vec<f32>,
    /// PCA basis P ∈ R^{dim x bits}, row-major. Projection: v' = (v-mean) · P.
    pub pca: Vec<f32>,
    pub rot: Box<dyn Rotation + Send + Sync>,
}

impl ItqEncoder {
    /// Fit PCA and rotation on `data` (n vectors of `dim` f32, row-major).
    pub fn fit(
        data: &[f32],
        dim: usize,
        bits: usize,
        kind: RotationKind,
        n_iter: usize,
        seed: u64,
    ) -> Self {
        assert!(bits > 0 && bits <= dim);
        assert_eq!(data.len() % dim, 0);
        let n = data.len() / dim;
        assert!(n >= bits, "need at least `bits` training rows");

        // 1. Centre the data.
        let mut mean = vec![0f32; dim];
        for row in data.chunks_exact(dim) {
            for (m, x) in mean.iter_mut().zip(row) { *m += *x; }
        }
        for m in &mut mean { *m /= n as f32; }

        let mut centred = Vec::with_capacity(data.len());
        for row in data.chunks_exact(dim) {
            for (x, m) in row.iter().zip(&mean) { centred.push(x - m); }
        }

        // 2. PCA basis via orthogonal iteration on X^T X.
        let pca = top_eigvecs(&centred, n, dim, bits, 25, seed ^ 0xA5A5_5A5A);

        // 3. Project all training rows into b-space.
        let mut v = vec![0f32; n * bits];
        for i in 0..n {
            for j in 0..bits {
                let mut s = 0.0f32;
                let x = &centred[i * dim..(i + 1) * dim];
                for k in 0..dim { s += x[k] * pca[k * bits + j]; }
                v[i * bits + j] = s;
            }
        }

        // 4. Build rotation.
        let rot: Box<dyn Rotation + Send + Sync> = match kind {
            RotationKind::Identity => Box::new(IdentityRotation),
            RotationKind::Random => {
                let r = random_orthogonal(bits, seed ^ 0xDEAD_BEEF);
                Box::new(DenseRotation::new("random", bits, r))
            }
            RotationKind::Itq => {
                let r = fit_itq(&v, n, bits, n_iter, seed ^ 0x1234_5678);
                Box::new(DenseRotation::new("itq", bits, r))
            }
        };

        Self { dim, bits, mean, pca, rot }
    }

    /// Encode one vector into packed bits (LSB = bit 0 of the b-vector).
    pub fn encode(&self, x: &[f32]) -> Vec<u64> {
        assert_eq!(x.len(), self.dim);
        let mut v = vec![0f32; self.bits];
        for j in 0..self.bits {
            let mut s = 0.0f32;
            for k in 0..self.dim {
                s += (x[k] - self.mean[k]) * self.pca[k * self.bits + j];
            }
            v[j] = s;
        }
        self.rot.apply(&mut v);
        pack_bits_from_signs(&v)
    }

    pub fn encode_many(&self, data: &[f32]) -> Vec<u64> {
        let n = data.len() / self.dim;
        let words = (self.bits + 63) / 64;
        let mut out = vec![0u64; n * words];
        for (i, row) in data.chunks_exact(self.dim).enumerate() {
            let code = self.encode(row);
            out[i * words..(i + 1) * words].copy_from_slice(&code);
        }
        out
    }

    /// Mean square quantisation error on the training rows (lower is better).
    pub fn training_mse(&self, data: &[f32]) -> f32 {
        let n = data.len() / self.dim;
        let mut err = 0.0f64;
        let mut v = vec![0f32; self.bits];
        for row in data.chunks_exact(self.dim) {
            for j in 0..self.bits {
                let mut s = 0.0f32;
                for k in 0..self.dim {
                    s += (row[k] - self.mean[k]) * self.pca[k * self.bits + j];
                }
                v[j] = s;
            }
            self.rot.apply(&mut v);
            for &x in &v {
                let b = if x >= 0.0 { 1.0 } else { -1.0 };
                let d = (x - b) as f64;
                err += d * d;
            }
        }
        (err / (n * self.bits) as f64) as f32
    }
}

// ---------- flat Hamming index ------------------------------------------------------

pub struct BinaryIndex {
    pub bits: usize,
    pub words: usize,
    pub n: usize,
    pub codes: Vec<u64>,
}

impl BinaryIndex {
    pub fn new(bits: usize, codes: Vec<u64>) -> Self {
        let words = (bits + 63) / 64;
        assert_eq!(codes.len() % words, 0);
        let n = codes.len() / words;
        Self { bits, words, n, codes }
    }

    pub fn search(&self, q: &[u64], k: usize) -> Vec<(u32, u32)> {
        assert_eq!(q.len(), self.words);
        // min-heap on distance ⇒ use a BinaryHeap with Reverse on distance.
        let mut heap: std::collections::BinaryHeap<(u32, u32)> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        for (idx, code) in self.codes.chunks_exact(self.words).enumerate() {
            let mut d: u32 = 0;
            for w in 0..self.words { d += (code[w] ^ q[w]).count_ones(); }
            if heap.len() < k {
                heap.push((d, idx as u32));
            } else if let Some(&(worst, _)) = heap.peek() {
                if d < worst {
                    heap.pop();
                    heap.push((d, idx as u32));
                }
            }
        }
        let mut out: Vec<(u32, u32)> = heap.into_iter().map(|(d, i)| (d, i)).collect();
        out.sort_by_key(|&(d, i)| (d, i));
        // return as (id, dist) for callers that want it other way:
        out.into_iter().map(|(d, i)| (i, d)).collect()
    }
}

// ---------- linear algebra helpers --------------------------------------------------

/// Top-`k` eigenvectors of X^T X via orthogonal (block power) iteration.
/// Returns P (dim x k) row-major. X is n x dim row-major.
fn top_eigvecs(x: &[f32], n: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut p = vec![0f32; dim * k];
    for v in &mut p { *v = rng.next_normal() as f32; }
    orthonormalise(&mut p, dim, k);

    // Reuse buffer for Y = X P (n x k) and Z = X^T Y (dim x k).
    let mut y = vec![0f32; n * k];
    let mut z = vec![0f32; dim * k];
    for _ in 0..iters {
        // Y = X P
        for i in 0..n {
            let row = &x[i * dim..(i + 1) * dim];
            let out = &mut y[i * k..(i + 1) * k];
            for j in 0..k {
                let mut s = 0.0f32;
                for d in 0..dim { s += row[d] * p[d * k + j]; }
                out[j] = s;
            }
        }
        // Z = X^T Y
        for d in 0..dim {
            for j in 0..k {
                let mut s = 0.0f32;
                for i in 0..n { s += x[i * dim + d] * y[i * k + j]; }
                z[d * k + j] = s;
            }
        }
        p.copy_from_slice(&z);
        orthonormalise(&mut p, dim, k);
    }
    p
}

/// Modified Gram-Schmidt on an `rows x cols` row-major matrix (cols = k ≤ rows).
fn orthonormalise(m: &mut [f32], rows: usize, cols: usize) {
    for j in 0..cols {
        // subtract projections of prior columns
        for i in 0..j {
            let mut dot = 0.0f32;
            for r in 0..rows { dot += m[r * cols + j] * m[r * cols + i]; }
            for r in 0..rows { m[r * cols + j] -= dot * m[r * cols + i]; }
        }
        // normalise
        let mut nrm = 0.0f32;
        for r in 0..rows { nrm += m[r * cols + j] * m[r * cols + j]; }
        let nrm = nrm.sqrt().max(1e-20);
        for r in 0..rows { m[r * cols + j] /= nrm; }
    }
}

/// Random b×b orthogonal matrix via QR on a Gaussian matrix.
fn random_orthogonal(b: usize, seed: u64) -> Vec<f32> {
    let mut rng = SplitMix64::new(seed);
    let mut m = vec![0f32; b * b];
    for v in &mut m { *v = rng.next_normal() as f32; }
    orthonormalise(&mut m, b, b);
    m
}

/// ITQ main loop. Returns the learned b×b rotation R (row-major).
fn fit_itq(v: &[f32], n: usize, b: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut r = random_orthogonal(b, seed);
    let mut vr = vec![0f32; n * b];
    let mut bmat = vec![0f32; n * b];
    for _ in 0..iters {
        // VR
        for i in 0..n {
            let vi = &v[i * b..(i + 1) * b];
            let out = &mut vr[i * b..(i + 1) * b];
            for j in 0..b {
                let mut s = 0.0f32;
                for k in 0..b { s += vi[k] * r[k * b + j]; }
                out[j] = s;
            }
        }
        // B = sign(VR)
        for i in 0..n * b {
            bmat[i] = if vr[i] >= 0.0 { 1.0 } else { -1.0 };
        }
        // M = V^T B  (b x b)
        let mut mmat = vec![0f32; b * b];
        for a in 0..b {
            for c in 0..b {
                let mut s = 0.0f32;
                for i in 0..n { s += v[i * b + a] * bmat[i * b + c]; }
                mmat[a * b + c] = s;
            }
        }
        // SVD: M = U Σ W^T  ⇒  R = W U^T
        let (u, _sigma, w) = jacobi_svd(&mmat, b);
        // r = W * U^T  (row-major)
        for i in 0..b {
            for j in 0..b {
                let mut s = 0.0f32;
                for t in 0..b { s += w[i * b + t] * u[j * b + t]; }
                r[i * b + j] = s;
            }
        }
    }
    r
}

/// One-sided Jacobi SVD of a square b×b matrix. Returns (U, Σ, W) with
/// A = U diag(Σ) W^T, all row-major. Small `b` only (≤ ~256).
fn jacobi_svd(a: &[f32], b: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Work in f64 for stability.
    let mut u: Vec<f64> = a.iter().map(|&x| x as f64).collect();
    let mut w = vec![0f64; b * b];
    for i in 0..b { w[i * b + i] = 1.0; }
    let tol: f64 = 1e-12 * (b as f64);
    for _sweep in 0..50 {
        let mut off: f64 = 0.0;
        for p in 0..b - 1 {
            for q in p + 1..b {
                // compute dot products of columns p and q of U
                let mut alpha = 0.0f64;
                let mut beta = 0.0f64;
                let mut gamma = 0.0f64;
                for i in 0..b {
                    let up = u[i * b + p];
                    let uq = u[i * b + q];
                    alpha += up * up;
                    beta  += uq * uq;
                    gamma += up * uq;
                }
                off += gamma * gamma;
                if gamma.abs() <= tol * (alpha.sqrt() * beta.sqrt() + 1e-300) { continue; }
                // Jacobi rotation angle
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta > 0.0 {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                } else {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..b {
                    let up = u[i * b + p];
                    let uq = u[i * b + q];
                    u[i * b + p] = c * up - s * uq;
                    u[i * b + q] = s * up + c * uq;
                    let wp = w[i * b + p];
                    let wq = w[i * b + q];
                    w[i * b + p] = c * wp - s * wq;
                    w[i * b + q] = s * wp + c * wq;
                }
            }
        }
        if off < tol * tol { break; }
    }
    // Column norms of U are the singular values. Normalise U.
    let mut sigma = vec![0f32; b];
    for j in 0..b {
        let mut nrm = 0.0f64;
        for i in 0..b { nrm += u[i * b + j] * u[i * b + j]; }
        let nrm = nrm.sqrt();
        sigma[j] = nrm as f32;
        if nrm > 1e-20 {
            for i in 0..b { u[i * b + j] /= nrm; }
        }
    }
    let u_f32: Vec<f32> = u.into_iter().map(|x| x as f32).collect();
    let w_f32: Vec<f32> = w.into_iter().map(|x| x as f32).collect();
    (u_f32, sigma, w_f32)
}

fn pack_bits_from_signs(v: &[f32]) -> Vec<u64> {
    let words = (v.len() + 63) / 64;
    let mut out = vec![0u64; words];
    for (i, &x) in v.iter().enumerate() {
        if x >= 0.0 { out[i / 64] |= 1u64 << (i % 64); }
    }
    out
}

// ---------- tiny deterministic RNG (SplitMix64) -------------------------------------

pub struct SplitMix64(u64);
impl SplitMix64 {
    pub fn new(seed: u64) -> Self { Self(seed.wrapping_add(0x9E3779B97F4A7C15)) }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    /// Box-Muller N(0,1).
    pub fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------- recall helper -----------------------------------------------------------

/// Brute-force L2 top-k ground truth. Returns ids only.
pub fn bruteforce_topk_l2(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut heap: std::collections::BinaryHeap<(OrderedF32, u32)> =
        std::collections::BinaryHeap::with_capacity(k + 1);
    for i in 0..n {
        let row = &data[i * dim..(i + 1) * dim];
        let mut d = 0.0f32;
        for k2 in 0..dim { let z = row[k2] - query[k2]; d += z * z; }
        if heap.len() < k {
            heap.push((OrderedF32(d), i as u32));
        } else if let Some(&(worst, _)) = heap.peek() {
            if d < worst.0 {
                heap.pop();
                heap.push((OrderedF32(d), i as u32));
            }
        }
    }
    let mut v: Vec<(OrderedF32, u32)> = heap.into_iter().collect();
    v.sort_by_key(|&(d, i)| (d, i));
    v.into_iter().map(|(_, i)| i).collect()
}

#[derive(Clone, Copy, PartialEq)]
pub struct OrderedF32(pub f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, o: &Self) -> Option<core::cmp::Ordering> { Some(self.cmp(o)) }
}
impl Ord for OrderedF32 {
    fn cmp(&self, o: &Self) -> core::cmp::Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(core::cmp::Ordering::Equal)
    }
}

pub fn recall_at_k(pred: &[u32], truth: &[u32]) -> f32 {
    use std::collections::HashSet;
    let t: HashSet<u32> = truth.iter().copied().collect();
    let hit = pred.iter().filter(|p| t.contains(p)).count();
    hit as f32 / truth.len() as f32
}

// placeholder to keep Reverse import used — downstream bin uses Reverse directly.
#[doc(hidden)]
pub fn _keep_reverse() -> Reverse<u32> { Reverse(0) }

// ---------- unit tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss_clusters(n: usize, dim: usize, nc: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut centres = vec![0f32; nc * dim];
        for v in &mut centres { *v = (rng.next_normal() as f32) * 3.0; }
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let c = (rng.next_u64() as usize) % nc;
            for d in 0..dim {
                out.push(centres[c * dim + d] + rng.next_normal() as f32);
            }
        }
        out
    }

    #[test]
    fn rotation_is_orthogonal() {
        let r = random_orthogonal(16, 42);
        // R R^T ≈ I
        for i in 0..16 {
            for j in 0..16 {
                let mut s = 0.0f32;
                for k in 0..16 { s += r[i * 16 + k] * r[j * 16 + k]; }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((s - want).abs() < 1e-3, "R R^T[{},{}]={}", i, j, s);
            }
        }
    }

    #[test]
    fn itq_reduces_training_mse_vs_identity() {
        let data = gauss_clusters(2000, 32, 8, 7);
        let e_id = ItqEncoder::fit(&data, 32, 16, RotationKind::Identity, 0, 7);
        let e_rand = ItqEncoder::fit(&data, 32, 16, RotationKind::Random, 0, 7);
        let e_itq = ItqEncoder::fit(&data, 32, 16, RotationKind::Itq, 40, 7);
        let m_id = e_id.training_mse(&data);
        let m_rand = e_rand.training_mse(&data);
        let m_itq = e_itq.training_mse(&data);
        println!("MSE identity={m_id:.4} random={m_rand:.4} itq={m_itq:.4}");
        assert!(m_itq < m_id * 0.99, "ITQ must beat identity (id={m_id}, itq={m_itq})");
        assert!(m_itq <= m_rand + 1e-3, "ITQ should not be worse than random (rand={m_rand}, itq={m_itq})");
    }

    #[test]
    fn hamming_search_finds_exact_self() {
        let data = gauss_clusters(500, 16, 4, 99);
        let enc = ItqEncoder::fit(&data, 16, 16, RotationKind::Itq, 30, 99);
        let codes = enc.encode_many(&data);
        let idx = BinaryIndex::new(16, codes);
        // self-query must return itself with distance 0 as the top result.
        let q = enc.encode(&data[..16]);
        let hits = idx.search(&q, 5);
        assert_eq!(hits[0].1, 0, "self-query distance must be zero");
        assert_eq!(hits[0].0, 0, "self-query top id must be 0");
    }

    #[test]
    fn binary_index_round_trip_dims() {
        let data = gauss_clusters(100, 24, 3, 1);
        let enc = ItqEncoder::fit(&data, 24, 24, RotationKind::Random, 0, 1);
        let codes = enc.encode_many(&data);
        let words = (enc.bits + 63) / 64;
        assert_eq!(codes.len(), 100 * words);
        let idx = BinaryIndex::new(enc.bits, codes);
        assert_eq!(idx.n, 100);
    }
}
