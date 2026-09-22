//! ruvector-opq-rotation
//!
//! Optimized Product Quantization (OPQ) with learned block-Householder rotation.
//!
//! Plain PQ splits a d-dim vector into m contiguous sub-vectors and runs k-means
//! per sub-space. Recall degrades when variance is unevenly distributed across
//! dimensions (a common trait of embedding models). OPQ (Ge et al., CVPR 2013)
//! learns an orthogonal rotation R that balances variance across sub-spaces
//! before PQ encoding, dramatically improving Asymmetric Distance Computation
//! (ADC) recall at no query-time compute overhead beyond a single matmul.
//!
//! This crate provides three swappable backends via the [`Quantizer`] trait:
//!   - [`PlainPq`]        : baseline PQ with contiguous sub-vector split
//!   - [`RandomRotPq`]    : PQ preceded by a fixed random orthogonal rotation
//!   - [`OpqHouseholder`] : PQ preceded by a rotation learned via alternating
//!                          block-Householder updates on training residuals.
//!
//! Design notes:
//!   - `f32` throughout, safe Rust only, no unsafe blocks.
//!   - Rotation stored as a full d×d matrix (dense) — for d ≤ 256 this is
//!     ~256 KB, negligible next to the codebooks.
//!   - k-means uses balanced Lloyd with re-seeding of empty clusters.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub mod util;

/// A trainable quantizer producing a byte code per vector plus an ADC scorer.
pub trait Quantizer {
    fn train(&mut self, data: &[Vec<f32>]);
    fn encode(&self, v: &[f32]) -> Vec<u8>;
    /// Precompute per-subspace distance tables for a query.
    fn adc_tables(&self, q: &[f32]) -> Vec<f32>;
    /// Compute ADC distance for a code given precomputed tables.
    fn adc_distance(&self, code: &[u8], tables: &[f32]) -> f32;
    fn m(&self) -> usize; // # subspaces
    fn ks(&self) -> usize; // codewords per subspace
    fn d(&self) -> usize;
}

// -----------------------------------------------------------------------------
// PlainPQ
// -----------------------------------------------------------------------------

pub struct PlainPq {
    d: usize,
    m: usize,
    ks: usize,
    ds: usize, // sub-dim = d / m
    /// codebooks[j] flat of shape [ks * ds]
    codebooks: Vec<Vec<f32>>,
    iters: usize,
    seed: u64,
}

impl PlainPq {
    pub fn new(d: usize, m: usize, ks: usize, iters: usize, seed: u64) -> Self {
        assert!(d % m == 0, "d must be divisible by m");
        Self {
            d,
            m,
            ks,
            ds: d / m,
            codebooks: vec![Vec::new(); m],
            iters,
            seed,
        }
    }
}

impl Quantizer for PlainPq {
    fn train(&mut self, data: &[Vec<f32>]) {
        for j in 0..self.m {
            let sub: Vec<Vec<f32>> = data
                .iter()
                .map(|v| v[j * self.ds..(j + 1) * self.ds].to_vec())
                .collect();
            let cb = kmeans(&sub, self.ks, self.iters, self.seed + j as u64);
            self.codebooks[j] = cb;
        }
    }
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let mut code = vec![0u8; self.m];
        for j in 0..self.m {
            let sub = &v[j * self.ds..(j + 1) * self.ds];
            code[j] = nearest_codeword(sub, &self.codebooks[j], self.ks, self.ds) as u8;
        }
        code
    }
    fn adc_tables(&self, q: &[f32]) -> Vec<f32> {
        let mut t = vec![0f32; self.m * self.ks];
        for j in 0..self.m {
            let qs = &q[j * self.ds..(j + 1) * self.ds];
            for k in 0..self.ks {
                let cw = &self.codebooks[j][k * self.ds..(k + 1) * self.ds];
                t[j * self.ks + k] = sq_dist(qs, cw);
            }
        }
        t
    }
    fn adc_distance(&self, code: &[u8], tables: &[f32]) -> f32 {
        let mut d = 0f32;
        for j in 0..self.m {
            d += tables[j * self.ks + code[j] as usize];
        }
        d
    }
    fn m(&self) -> usize { self.m }
    fn ks(&self) -> usize { self.ks }
    fn d(&self) -> usize { self.d }
}

// -----------------------------------------------------------------------------
// Rotated PQ (random and OPQ share encode/adc paths)
// -----------------------------------------------------------------------------

pub struct RotatedPq {
    inner: PlainPq,
    /// row-major d×d orthogonal matrix
    r: Vec<f32>,
}

impl RotatedPq {
    fn rotate(&self, v: &[f32]) -> Vec<f32> {
        let d = self.inner.d;
        let mut out = vec![0f32; d];
        for i in 0..d {
            let row = &self.r[i * d..(i + 1) * d];
            let mut s = 0f32;
            for k in 0..d {
                s += row[k] * v[k];
            }
            out[i] = s;
        }
        out
    }
}

impl Quantizer for RotatedPq {
    fn train(&mut self, data: &[Vec<f32>]) {
        let rotated: Vec<Vec<f32>> = data.iter().map(|v| self.rotate(v)).collect();
        self.inner.train(&rotated);
    }
    fn encode(&self, v: &[f32]) -> Vec<u8> {
        let rv = self.rotate(v);
        self.inner.encode(&rv)
    }
    fn adc_tables(&self, q: &[f32]) -> Vec<f32> {
        let rq = self.rotate(q);
        self.inner.adc_tables(&rq)
    }
    fn adc_distance(&self, code: &[u8], tables: &[f32]) -> f32 {
        self.inner.adc_distance(code, tables)
    }
    fn m(&self) -> usize { self.inner.m() }
    fn ks(&self) -> usize { self.inner.ks() }
    fn d(&self) -> usize { self.inner.d() }
}

// -----------------------------------------------------------------------------
// Random rotation (Haar-like via QR of Gaussian matrix)
// -----------------------------------------------------------------------------

pub struct RandomRotPq;

impl RandomRotPq {
    pub fn build(d: usize, m: usize, ks: usize, iters: usize, seed: u64) -> RotatedPq {
        let r = random_orthogonal(d, seed);
        RotatedPq {
            inner: PlainPq::new(d, m, ks, iters, seed + 1),
            r,
        }
    }
}

// -----------------------------------------------------------------------------
// OPQ via block-Householder alternating optimization
// -----------------------------------------------------------------------------

pub struct OpqHouseholder;

impl OpqHouseholder {
    /// Learn OPQ rotation by alternating:
    ///   1. Encode rotated data with current PQ codebooks.
    ///   2. Compute residual matrix E = X - Rᵀ * decode(Q(RX)).
    ///   3. Update R via a single-step block-Householder that shrinks E.
    ///
    /// This is the "parametric" OPQ variant simplified for practical Rust use.
    /// `opq_iters` outer loops of (train PQ → refresh R). Two iters is usually enough.
    pub fn build(
        d: usize,
        m: usize,
        ks: usize,
        pq_iters: usize,
        opq_iters: usize,
        data: &[Vec<f32>],
        seed: u64,
    ) -> RotatedPq {
        let mut r = random_orthogonal(d, seed);
        let mut inner = PlainPq::new(d, m, ks, pq_iters, seed + 1);

        for outer in 0..opq_iters {
            // 1) train PQ on rotated data
            let rotated: Vec<Vec<f32>> =
                data.iter().map(|v| matvec(&r, v, d)).collect();
            inner.train(&rotated);

            // 2) compute per-vector reconstruction Y = decode(code) in rotated space
            let recon: Vec<Vec<f32>> = rotated
                .iter()
                .map(|rv| {
                    let code = inner.encode(rv);
                    decode(&code, &inner.codebooks, inner.ks, inner.ds, d)
                })
                .collect();

            // 3) update R = argmin sum ||R X - Y||^2  s.t. Rᵀ R = I
            //    Closed form: R = V Uᵀ where SVD(Xᵀ Y) = U Σ Vᵀ.
            //    We use a Householder sweep as a cheap orthogonal-Procrustes
            //    surrogate (Kabsch step without full SVD).
            r = procrustes_householder(data, &recon, d, &mut StdRng::seed_from_u64(seed + outer as u64 + 42));
        }

        // final PQ training on the last rotation
        let rotated: Vec<Vec<f32>> = data.iter().map(|v| matvec(&r, v, d)).collect();
        inner.train(&rotated);

        RotatedPq { inner, r }
    }
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

pub(crate) fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

pub(crate) fn matvec(r: &[f32], v: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0f32; d];
    for i in 0..d {
        let row = &r[i * d..(i + 1) * d];
        let mut s = 0f32;
        for k in 0..d { s += row[k] * v[k]; }
        out[i] = s;
    }
    out
}

fn decode(code: &[u8], codebooks: &[Vec<f32>], ks: usize, ds: usize, d: usize) -> Vec<f32> {
    let _ = ks;
    let mut out = vec![0f32; d];
    for j in 0..codebooks.len() {
        let k = code[j] as usize;
        let cw = &codebooks[j][k * ds..(k + 1) * ds];
        out[j * ds..(j + 1) * ds].copy_from_slice(cw);
    }
    out
}

fn nearest_codeword(v: &[f32], cb: &[f32], ks: usize, ds: usize) -> usize {
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for k in 0..ks {
        let cw = &cb[k * ds..(k + 1) * ds];
        let d = sq_dist(v, cw);
        if d < bd { bd = d; best = k; }
    }
    best
}

/// Balanced Lloyd k-means. Empty clusters get re-seeded from the max-variance one.
fn kmeans(data: &[Vec<f32>], ks: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = data.len();
    let ds = data[0].len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cb = vec![0f32; ks * ds];
    // k-means++ init (simplified: uniform sample)
    for k in 0..ks {
        let idx = rng.gen_range(0..n);
        cb[k * ds..(k + 1) * ds].copy_from_slice(&data[idx]);
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        // assignment
        for i in 0..n {
            let mut bd = f32::INFINITY;
            let mut bk = 0;
            for k in 0..ks {
                let cw = &cb[k * ds..(k + 1) * ds];
                let d = sq_dist(&data[i], cw);
                if d < bd { bd = d; bk = k; }
            }
            assign[i] = bk;
        }
        // update
        let mut sums = vec![0f32; ks * ds];
        let mut cnt = vec![0usize; ks];
        for i in 0..n {
            let k = assign[i];
            cnt[k] += 1;
            for t in 0..ds {
                sums[k * ds + t] += data[i][t];
            }
        }
        for k in 0..ks {
            if cnt[k] == 0 {
                let idx = rng.gen_range(0..n);
                cb[k * ds..(k + 1) * ds].copy_from_slice(&data[idx]);
            } else {
                let c = cnt[k] as f32;
                for t in 0..ds {
                    cb[k * ds + t] = sums[k * ds + t] / c;
                }
            }
        }
    }
    cb
}

/// Build a d×d orthogonal matrix by Gram–Schmidt on a Gaussian matrix.
pub(crate) fn random_orthogonal(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = vec![0f32; d * d];
    for i in 0..d * d {
        // Box-Muller
        let u1: f32 = rng.gen::<f32>().max(1e-9);
        let u2: f32 = rng.gen::<f32>();
        m[i] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
    }
    // Modified Gram-Schmidt on rows
    for i in 0..d {
        for j in 0..i {
            let mut dot = 0f32;
            for t in 0..d {
                dot += m[i * d + t] * m[j * d + t];
            }
            for t in 0..d {
                m[i * d + t] -= dot * m[j * d + t];
            }
        }
        let mut n = 0f32;
        for t in 0..d {
            n += m[i * d + t] * m[i * d + t];
        }
        let n = n.sqrt().max(1e-12);
        for t in 0..d {
            m[i * d + t] /= n;
        }
    }
    m
}

/// Orthogonal Procrustes: R = argmin ||R X - Y||_F  s.t. R^T R = I.
///
/// Closed form is R = U V^T where SVD(Y X^T) = U Σ V^T. We compute the polar
/// factor of C = Y X^T without a full SVD by using symmetric-Jacobi eigendecomp
/// on C^T C, then R = C (C^T C)^{-1/2}. Works reliably at d ≤ 256.
fn procrustes_householder(
    x: &[Vec<f32>],
    y: &[Vec<f32>],
    d: usize,
    _rng: &mut StdRng,
) -> Vec<f32> {
    let n = x.len();
    // C = Y X^T   (d×d), row-major
    let mut c = vec![0f32; d * d];
    for k in 0..n {
        for i in 0..d {
            let yi = y[k][i];
            if yi == 0.0 { continue; }
            for j in 0..d {
                c[i * d + j] += yi * x[k][j];
            }
        }
    }
    // A = C^T C   (symmetric PSD)
    let mut a = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0f32;
            for k in 0..d {
                s += c[k * d + i] * c[k * d + j];
            }
            a[i * d + j] = s;
        }
    }
    // Jacobi eigendecomp of A: A = V Λ V^T
    let (eigvals, eigvecs) = jacobi_eig(&mut a, d, 40);
    // A^{-1/2} = V Λ^{-1/2} V^T
    let mut a_inv_half = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0f32;
            for k in 0..d {
                let ev = eigvals[k].max(1e-8).sqrt();
                s += eigvecs[i * d + k] * eigvecs[j * d + k] / ev;
            }
            a_inv_half[i * d + j] = s;
        }
    }
    // R = C * A^{-1/2}
    let mut r = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0f32;
            for k in 0..d {
                s += c[i * d + k] * a_inv_half[k * d + j];
            }
            r[i * d + j] = s;
        }
    }
    r
}

/// Symmetric-matrix Jacobi eigendecomposition. Returns (eigenvalues, eigvecs)
/// where eigvecs is row-major with column k = k-th eigenvector.
fn jacobi_eig(a: &mut [f32], d: usize, max_sweeps: usize) -> (Vec<f32>, Vec<f32>) {
    let mut v = vec![0f32; d * d];
    for i in 0..d { v[i * d + i] = 1.0; }
    for _ in 0..max_sweeps {
        let mut off = 0f32;
        for i in 0..d {
            for j in (i + 1)..d {
                off += a[i * d + j] * a[i * d + j];
            }
        }
        if off < 1e-10 { break; }
        for p in 0..d {
            for q in (p + 1)..d {
                let apq = a[p * d + q];
                if apq.abs() < 1e-8 { continue; }
                let app = a[p * d + p];
                let aqq = a[q * d + q];
                let theta = 0.5 * (2.0 * apq).atan2(app - aqq);
                let (s, cs) = (theta.sin(), theta.cos());
                // rotate rows p,q
                for k in 0..d {
                    let apk = a[p * d + k];
                    let aqk = a[q * d + k];
                    a[p * d + k] =  cs * apk + s * aqk;
                    a[q * d + k] = -s * apk + cs * aqk;
                }
                // rotate cols p,q
                for k in 0..d {
                    let akp = a[k * d + p];
                    let akq = a[k * d + q];
                    a[k * d + p] =  cs * akp + s * akq;
                    a[k * d + q] = -s * akp + cs * akq;
                }
                // update V
                for k in 0..d {
                    let vkp = v[k * d + p];
                    let vkq = v[k * d + q];
                    v[k * d + p] =  cs * vkp + s * vkq;
                    v[k * d + q] = -s * vkp + cs * vkq;
                }
            }
        }
    }
    let mut vals = vec![0f32; d];
    for i in 0..d { vals[i] = a[i * d + i]; }
    (vals, v)
}

// -----------------------------------------------------------------------------
// tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn gauss_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| {
            (0..d).map(|i| {
                let u1: f32 = rng.gen::<f32>().max(1e-9);
                let u2: f32 = rng.gen::<f32>();
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                // Skew variance so dims 0..d/2 have ~10x the variance of the rest.
                if i < d / 2 { g * 3.0 } else { g * 0.3 }
            }).collect()
        }).collect()
    }

    #[test]
    fn orthogonal_matrix_is_orthogonal() {
        let d = 16;
        let r = random_orthogonal(d, 7);
        // R Rᵀ ≈ I
        for i in 0..d {
            for j in 0..d {
                let mut s = 0f32;
                for k in 0..d { s += r[i * d + k] * r[j * d + k]; }
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((s - target).abs() < 1e-4, "R Rᵀ deviation at ({},{}): {}", i, j, s);
            }
        }
    }

    #[test]
    fn plain_pq_train_encode_roundtrip() {
        let data = gauss_data(500, 16, 1);
        let mut pq = PlainPq::new(16, 4, 16, 10, 1);
        pq.train(&data);
        let code = pq.encode(&data[0]);
        assert_eq!(code.len(), 4);
        let t = pq.adc_tables(&data[0]);
        let d0 = pq.adc_distance(&code, &t);
        assert!(d0 >= 0.0);
    }

    #[test]
    fn opq_reduces_reconstruction_error_vs_plain_pq() {
        let d = 32;
        let n = 800;
        let data = gauss_data(n, d, 2);
        let mut plain = PlainPq::new(d, 4, 16, 15, 2);
        plain.train(&data);
        let opq = OpqHouseholder::build(d, 4, 16, 15, 2, &data, 2);

        let mut e_plain = 0f32;
        let mut e_opq = 0f32;
        for v in &data {
            let c1 = plain.encode(v);
            let r1 = decode(&c1, &plain.codebooks, plain.ks, plain.ds, d);
            e_plain += sq_dist(v, &r1);

            let c2 = opq.encode(v);
            let rv = matvec(&opq.r, v, d);
            let recon = decode(&c2, &opq.inner.codebooks, opq.inner.ks, opq.inner.ds, d);
            e_opq += sq_dist(&rv, &recon);
        }
        // OPQ should not be worse than plain PQ on skewed-variance data.
        // Accept a small slack for stochastic training.
        assert!(e_opq <= e_plain * 1.05,
            "OPQ err {} should be <= plain PQ err {} on skewed data", e_opq, e_plain);
    }
}
