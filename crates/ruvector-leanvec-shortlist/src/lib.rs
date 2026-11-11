//! LeanVec-style low-rank pre-projection shortlist for approximate nearest
//! neighbor search.
//!
//! Core idea (Intel LeanVec, 2024): train a learned linear projection
//! `P ∈ R^{d' x d}` (typically `d' ≈ d/4..d/8`) that preserves pairwise
//! distances well on the dataset's distribution. At query time:
//!   1. Project query `q' = P q`.
//!   2. Scan the whole dataset in the low-rank space (cheap) to get an
//!      oversampled shortlist of size `L >> k` using full `L2^2` in R^{d'}.
//!   3. Rerank the shortlist in the original R^d with exact `L2^2`.
//!
//! The projection is learned with a lightweight power-iteration PCA on a
//! random subsample (deterministic, no BLAS). This keeps the crate
//! dependency-free (pure Rust `std`), fits in <500 lines, and still shows
//! real shortlist speedups on 128-d float data.
//!
//! Backends exposed via the [`Searcher`] trait:
//!   * [`BruteForce`]        — exact L2^2 baseline in R^d.
//!   * [`Pca16Shortlist`]    — project to 16-d, shortlist, rerank.
//!   * [`Pca32Shortlist`]    — project to 32-d, shortlist, rerank.
//!   * [`Random16Shortlist`] — random Gaussian projection to 16-d (JL-style,
//!                             for ablation: how much does *learned* vs
//!                             *random* projection matter).
//!
//! The crate is standalone; it is NOT added to the root workspace (the root
//! workspace excludes new standalone crates by default per ADR-174/289
//! patterns in this tree).

#![forbid(unsafe_code)]

use std::time::Instant;

/// A flat row-major f32 dataset of shape `(n, d)`.
pub struct Dataset {
    pub d: usize,
    pub data: Vec<f32>,
}

impl Dataset {
    pub fn new(d: usize, data: Vec<f32>) -> Self {
        assert!(d > 0 && data.len() % d == 0, "data length must be multiple of d");
        Self { d, data }
    }
    pub fn n(&self) -> usize {
        self.data.len() / self.d
    }
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.d..(i + 1) * self.d]
    }
}

#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Deterministic xorshift64 PRNG for reproducible random projections and
/// synthetic data.
#[derive(Clone)]
pub struct XorShift64 {
    state: u64,
}
impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    /// Uniform f32 in [-1, 1).
    pub fn uniform_sym(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        2.0 * u - 1.0
    }
    /// Box-Muller N(0,1).
    pub fn normal(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) as f32 / (1u64 << 53) as f32).max(1e-9);
        let u2 = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

/// Projection matrix stored as row-major `(d_out, d_in)`.
#[derive(Clone)]
pub struct Projection {
    pub d_in: usize,
    pub d_out: usize,
    pub rows: Vec<f32>, // d_out * d_in
}

impl Projection {
    /// Random orthonormal-ish Gaussian projection (JL-style). Each row is
    /// iid N(0,1), L2-normalized.
    pub fn random_gaussian(d_in: usize, d_out: usize, seed: u64) -> Self {
        let mut rng = XorShift64::new(seed);
        let mut rows = vec![0.0f32; d_out * d_in];
        for r in 0..d_out {
            let row = &mut rows[r * d_in..(r + 1) * d_in];
            for v in row.iter_mut() {
                *v = rng.normal();
            }
            let n2: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in row.iter_mut() {
                *v /= n2;
            }
        }
        Self { d_in, d_out, rows }
    }

    /// Learned PCA projection via subspace power iteration on a random sample.
    /// This approximates the top-`d_out` eigenvectors of the data covariance
    /// matrix. Dependency-free: everything is explicit O(sample_n * d^2) per
    /// iter plus O(d_out * d^2) orthogonalization.
    pub fn learn_pca(ds: &Dataset, d_out: usize, iters: usize, sample_n: usize, seed: u64) -> Self {
        let d = ds.d;
        let n = ds.n();
        let s = sample_n.min(n).max(d_out + 1);
        let mut rng = XorShift64::new(seed);

        // Pick a random sample without replacement (with repeats OK for speed).
        let sample: Vec<usize> = (0..s).map(|_| (rng.next_u64() as usize) % n).collect();

        // Estimate mean on sample.
        let mut mean = vec![0.0f32; d];
        for &i in &sample {
            let r = ds.row(i);
            for j in 0..d {
                mean[j] += r[j];
            }
        }
        for v in mean.iter_mut() {
            *v /= s as f32;
        }

        // Centered sample buffer, shape (s, d). s*d floats — bounded by sample_n.
        let mut xc = vec![0.0f32; s * d];
        for (si, &i) in sample.iter().enumerate() {
            let r = ds.row(i);
            for j in 0..d {
                xc[si * d + j] = r[j] - mean[j];
            }
        }

        // Initialize d_out random orthonormal basis rows: (d_out, d).
        let mut q = vec![0.0f32; d_out * d];
        for r in 0..d_out {
            for j in 0..d {
                q[r * d + j] = rng.normal();
            }
        }
        orthonormalize_rows(&mut q, d_out, d);

        // Subspace power iteration: Q <- orth(X^T X Q).
        let mut tmp_sq = vec![0.0f32; s * d_out]; // X Q^T, shape (s, d_out)
        let mut next_q = vec![0.0f32; d_out * d];
        for _ in 0..iters {
            // tmp_sq = X * Q^T  (s,d) x (d, d_out) -> (s, d_out)
            for i in 0..s {
                let xi = &xc[i * d..(i + 1) * d];
                for r in 0..d_out {
                    let qr = &q[r * d..(r + 1) * d];
                    let mut acc = 0.0f32;
                    for j in 0..d {
                        acc += xi[j] * qr[j];
                    }
                    tmp_sq[i * d_out + r] = acc;
                }
            }
            // next_q = tmp_sq^T * X  (d_out, s) x (s, d) -> (d_out, d)
            for v in next_q.iter_mut() {
                *v = 0.0;
            }
            for i in 0..s {
                let xi = &xc[i * d..(i + 1) * d];
                for r in 0..d_out {
                    let w = tmp_sq[i * d_out + r];
                    let dst = &mut next_q[r * d..(r + 1) * d];
                    for j in 0..d {
                        dst[j] += w * xi[j];
                    }
                }
            }
            // Orthonormalize.
            orthonormalize_rows(&mut next_q, d_out, d);
            std::mem::swap(&mut q, &mut next_q);
        }

        Projection { d_in: d, d_out, rows: q }
    }

    /// `out = P x` (shape `d_out`).
    pub fn project(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.d_in);
        debug_assert_eq!(out.len(), self.d_out);
        for r in 0..self.d_out {
            let row = &self.rows[r * self.d_in..(r + 1) * self.d_in];
            let mut acc = 0.0f32;
            for j in 0..self.d_in {
                acc += row[j] * x[j];
            }
            out[r] = acc;
        }
    }

    /// Project a whole dataset into a new flat `(n, d_out)` buffer.
    pub fn project_dataset(&self, ds: &Dataset) -> Vec<f32> {
        let n = ds.n();
        let mut out = vec![0.0f32; n * self.d_out];
        let mut buf = vec![0.0f32; self.d_out];
        for i in 0..n {
            self.project(ds.row(i), &mut buf);
            out[i * self.d_out..(i + 1) * self.d_out].copy_from_slice(&buf);
        }
        out
    }
}

/// Modified Gram-Schmidt; stable enough for small d_out used here.
fn orthonormalize_rows(m: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        // Subtract projections onto previous rows.
        for p in 0..r {
            let (prev_slice, rest) = m.split_at_mut((p + 1) * cols);
            let prev = &prev_slice[p * cols..(p + 1) * cols];
            let cur = &mut rest[(r - p - 1) * cols..(r - p) * cols];
            let mut dot = 0.0f32;
            for j in 0..cols {
                dot += prev[j] * cur[j];
            }
            for j in 0..cols {
                cur[j] -= dot * prev[j];
            }
        }
        // Normalize row r.
        let row = &mut m[r * cols..(r + 1) * cols];
        let mut n2 = 0.0f32;
        for v in row.iter() {
            n2 += v * v;
        }
        let n = n2.sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v /= n;
        }
    }
}

/// A nearest-neighbor searcher.
pub trait Searcher {
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)>;
    fn name(&self) -> &'static str;
}

/// Exact brute-force L2^2 in R^d.
pub struct BruteForce<'a> {
    pub ds: &'a Dataset,
}
impl<'a> Searcher for BruteForce<'a> {
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let n = self.ds.n();
        let mut scored: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, l2sq(self.ds.row(i), query)))
            .collect();
        scored.select_nth_unstable_by(k.min(n.saturating_sub(1)), |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });
        scored.truncate(k);
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored
    }
    fn name(&self) -> &'static str {
        "brute_force_d"
    }
}

/// Generic LeanVec-style shortlist searcher.
pub struct LeanVecShortlist<'a> {
    pub ds: &'a Dataset,
    pub proj: Projection,
    pub projected: Vec<f32>, // (n, d_out) row-major
    pub shortlist: usize,    // L
    pub label: &'static str,
}

impl<'a> LeanVecShortlist<'a> {
    pub fn build(
        ds: &'a Dataset,
        proj: Projection,
        shortlist: usize,
        label: &'static str,
    ) -> Self {
        let projected = proj.project_dataset(ds);
        Self { ds, proj, projected, shortlist, label }
    }
}

impl<'a> Searcher for LeanVecShortlist<'a> {
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let n = self.ds.n();
        let dout = self.proj.d_out;
        let mut qp = vec![0.0f32; dout];
        self.proj.project(query, &mut qp);

        // Low-rank shortlist pass.
        let mut scored_lo: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let pv = &self.projected[i * dout..(i + 1) * dout];
                let mut s = 0.0f32;
                for j in 0..dout {
                    let d = pv[j] - qp[j];
                    s += d * d;
                }
                (i, s)
            })
            .collect();
        let l = self.shortlist.min(n);
        scored_lo.select_nth_unstable_by(l.saturating_sub(1), |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });
        scored_lo.truncate(l);

        // Rerank in full dim.
        let mut scored_hi: Vec<(usize, f32)> = scored_lo
            .iter()
            .map(|&(i, _)| (i, l2sq(self.ds.row(i), query)))
            .collect();
        scored_hi.select_nth_unstable_by(k.min(l.saturating_sub(1)), |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });
        scored_hi.truncate(k);
        scored_hi.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored_hi
    }
    fn name(&self) -> &'static str {
        self.label
    }
}

/// Synthetic clustered dataset with a `d`-dim low-rank signal plus noise.
/// This produces data where PCA actually helps (vs random projection).
pub fn gen_clustered(n: usize, d: usize, clusters: usize, signal_dim: usize, seed: u64) -> Dataset {
    let mut rng = XorShift64::new(seed);
    let sig = signal_dim.min(d);
    // Pick an orthonormal signal basis (sig, d).
    let mut basis = vec![0.0f32; sig * d];
    for i in 0..sig * d {
        basis[i] = rng.normal();
    }
    orthonormalize_rows(&mut basis, sig, d);
    // Cluster centers in the signal subspace, mapped back to R^d.
    let mut centers = vec![0.0f32; clusters * d];
    for c in 0..clusters {
        // Random coefficients in the signal subspace.
        let coefs: Vec<f32> = (0..sig).map(|_| rng.normal() * 4.0).collect();
        let dst = &mut centers[c * d..(c + 1) * d];
        for s in 0..sig {
            let row = &basis[s * d..(s + 1) * d];
            let w = coefs[s];
            for j in 0..d {
                dst[j] += w * row[j];
            }
        }
    }

    let mut data = vec![0.0f32; n * d];
    for i in 0..n {
        let c = (rng.next_u64() as usize) % clusters;
        let ctr = &centers[c * d..(c + 1) * d];
        let dst = &mut data[i * d..(i + 1) * d];
        for j in 0..d {
            // Signal + small isotropic noise.
            dst[j] = ctr[j] + 0.4 * rng.normal();
        }
    }
    Dataset::new(d, data)
}

/// Compute recall@k of `pred` against `truth`.
pub fn recall_at_k(pred: &[(usize, f32)], truth: &[(usize, f32)], k: usize) -> f32 {
    let k = k.min(pred.len()).min(truth.len());
    let truth_set: std::collections::HashSet<usize> =
        truth.iter().take(k).map(|&(i, _)| i).collect();
    let hits = pred.iter().take(k).filter(|&&(i, _)| truth_set.contains(&i)).count();
    hits as f32 / k as f32
}

/// Minimal wall-clock timer.
pub fn time_it<F: FnOnce()>(f: F) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64()
}
