//! Anisotropic Product Quantization (APQ) for ruvector.
//!
//! Implements three swappable PQ variants behind a single [`Quantizer`] trait:
//!
//! * [`PlainPQ`]      — classic Lloyd's k-means per subspace (baseline).
//! * [`AnisotropicPQ`] — ScaNN-style score-aware loss that upweights the
//!   component of residual error that is *parallel* to the datapoint
//!   (Guo et al., ICML 2020, "Accelerating Large-Scale Inference with
//!   Anisotropic Vector Quantization"). Parallel error hurts inner-product /
//!   score preservation much more than orthogonal error; APQ trains PQ
//!   codebooks that explicitly bound the parallel term.
//! * [`AnisotropicPQR`] — APQ preceded by a learned orthogonal rotation
//!   (OPQ-non-parametric: eigen-basis of the data covariance) that spreads
//!   variance evenly across PQ subspaces, further reducing MSE.
//!
//! All three implement the same [`Quantizer`] trait, so downstream ruvector
//! crates (IVF, HNSW-rerank, DiskANN) can swap the backend without changing
//! call sites.
//!
//! ## Design notes
//!
//! * Pure safe Rust; no BLAS, no unsafe, no external heavy deps.
//! * The k-means uses k-means++ init and a fixed iteration budget so runs are
//!   deterministic given the seed and reproducible in CI.
//! * Anisotropic k-means uses the closed-form weighted update from §3.3 of
//!   Guo et al.: each cluster's centroid is the h-weighted mean of its
//!   members, where the h-weights come from splitting each residual into
//!   parallel and orthogonal parts w.r.t. the datapoint direction.
//! * Distance at query time is *symmetric* PQ (SDC) using a precomputed
//!   `k x k` inter-centroid table per subspace — this keeps the benchmark
//!   focused on codebook quality rather than table-computation tricks.

#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::many_single_char_names)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use thiserror::Error;

pub mod kmeans;
pub mod rotation;

/// Errors raised while training or encoding.
#[derive(Debug, Error)]
pub enum PqError {
    #[error("dimension {dim} is not divisible by number of subspaces {m}")]
    BadShape { dim: usize, m: usize },
    #[error("k ({k}) must be > 0 and <= 256")]
    BadK { k: usize },
    #[error("empty training set")]
    EmptyTraining,
    #[error("query dim {got} != trained dim {expected}")]
    DimMismatch { got: usize, expected: usize },
}

/// One-byte PQ code: `m` bytes, one per subspace (k must be ≤ 256).
pub type Code = Vec<u8>;

/// Uniform interface for PQ variants.
pub trait Quantizer {
    fn dim(&self) -> usize;
    fn m(&self) -> usize;
    fn k(&self) -> usize;

    /// Encode a single vector to its PQ code.
    fn encode(&self, x: &[f32]) -> Result<Code, PqError>;

    /// Symmetric squared L2 distance between two PQ codes.
    fn sdc(&self, a: &Code, b: &Code) -> f32;

    /// Reconstruct an approximate vector from a code (useful for reranking).
    fn reconstruct(&self, code: &Code) -> Vec<f32>;

    /// Bytes on disk per encoded vector.
    fn bytes_per_code(&self) -> usize {
        self.m()
    }
}

/// Shared PQ layout parameters.
#[derive(Debug, Clone, Copy)]
pub struct PqParams {
    /// Ambient dimension.
    pub dim: usize,
    /// Number of subspaces (`dim % m == 0`).
    pub m: usize,
    /// Centroids per subspace (≤ 256 so the code byte fits in u8).
    pub k: usize,
    /// k-means iterations per subspace.
    pub iters: usize,
    /// RNG seed for k-means++ + deterministic tie-breaks.
    pub seed: u64,
}

impl PqParams {
    pub fn ds(&self) -> usize {
        self.dim / self.m
    }
    pub fn validate(&self) -> Result<(), PqError> {
        if self.dim == 0 || self.dim % self.m != 0 {
            return Err(PqError::BadShape {
                dim: self.dim,
                m: self.m,
            });
        }
        if self.k == 0 || self.k > 256 {
            return Err(PqError::BadK { k: self.k });
        }
        Ok(())
    }
}

// ============================================================================
// PlainPQ — Lloyd's k-means baseline.
// ============================================================================

pub struct PlainPQ {
    params: PqParams,
    /// Codebooks: `m` subspaces × `k` centroids × `ds` floats.
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Precomputed `k*k` SDC table per subspace (squared L2).
    sdc_tables: Vec<Vec<f32>>,
}

impl PlainPQ {
    pub fn train(train: &[Vec<f32>], params: PqParams) -> Result<Self, PqError> {
        params.validate()?;
        if train.is_empty() {
            return Err(PqError::EmptyTraining);
        }
        let ds = params.ds();
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(params.m);
        for si in 0..params.m {
            let sub: Vec<Vec<f32>> = train
                .iter()
                .map(|v| v[si * ds..(si + 1) * ds].to_vec())
                .collect();
            let cents = kmeans::lloyd(&sub, params.k, params.iters, params.seed ^ (si as u64));
            codebooks.push(cents);
        }
        let sdc_tables = build_sdc_tables(&codebooks);
        Ok(Self {
            params,
            codebooks,
            sdc_tables,
        })
    }
}

impl Quantizer for PlainPQ {
    fn dim(&self) -> usize {
        self.params.dim
    }
    fn m(&self) -> usize {
        self.params.m
    }
    fn k(&self) -> usize {
        self.params.k
    }
    fn encode(&self, x: &[f32]) -> Result<Code, PqError> {
        encode_generic(x, &self.codebooks, &self.params)
    }
    fn sdc(&self, a: &Code, b: &Code) -> f32 {
        sdc_lookup(a, b, &self.sdc_tables, self.params.k)
    }
    fn reconstruct(&self, code: &Code) -> Vec<f32> {
        reconstruct_generic(code, &self.codebooks, &self.params)
    }
}

// ============================================================================
// AnisotropicPQ — ScaNN-style score-aware k-means.
// ============================================================================

/// Anisotropic k-means fit for one full-dim vector. We follow the
/// "per-vector parallel/orthogonal weight" formulation of Guo et al. §3.3.
///
/// For a residual `r = x - c` where `c` is the concatenated centroid:
///
/// * `r_parallel = ((r·x) / ||x||^2) * x`      (score-affecting part)
/// * `r_orth     = r - r_parallel`
///
/// Anisotropic loss: `w_par * ||r_par||^2 + w_orth * ||r_orth||^2`.
/// With `w_par >= w_orth`, k-means is pulled toward centroids that reduce
/// the parallel residual — exactly what MIPS/cosine ranking cares about.
///
/// We reduce the full-dim weighted problem to a per-subspace weighted
/// k-means by projecting the per-vector direction into each subspace and
/// using the *same* h-weights (an approximation that Guo et al. show works
/// well and, importantly, keeps the per-subspace loop O(N·k·ds)).
pub struct AnisotropicPQ {
    params: PqParams,
    /// `w_par / w_orth`. `1.0` recovers PlainPQ (up to init noise).
    eta: f32,
    codebooks: Vec<Vec<Vec<f32>>>,
    sdc_tables: Vec<Vec<f32>>,
}

impl AnisotropicPQ {
    pub fn train(train: &[Vec<f32>], params: PqParams, eta: f32) -> Result<Self, PqError> {
        params.validate()?;
        if train.is_empty() {
            return Err(PqError::EmptyTraining);
        }
        let ds = params.ds();

        // Per-vector parallel/orthogonal weights derived from ||x||.
        // Following Guo et al. Prop. 3.4, for unit-norm data the optimal
        // ratio grows with target compression; we expose `eta` directly so
        // callers can sweep it in benchmarks.
        let norms: Vec<f32> = train
            .iter()
            .map(|v| v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12))
            .collect();

        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(params.m);
        for si in 0..params.m {
            let sub: Vec<Vec<f32>> = train
                .iter()
                .map(|v| v[si * ds..(si + 1) * ds].to_vec())
                .collect();
            // Sub-vector direction for the parallel projection *within this
            // subspace* — equivalent to projecting the full-dim direction
            // onto the subspace basis, since PQ subspaces are axis-aligned.
            // Direction restricted to this subspace is the sub-slice of the
            // full-vector unit direction: (x_sub) / ||x_full||. NOT
            // normalized within the subspace — that would inflate parallel
            // weight for subspaces that happen to have small local energy.
            let sub_dirs: Vec<Vec<f32>> = train
                .iter()
                .zip(norms.iter())
                .map(|(v, n)| {
                    let inv = 1.0 / *n;
                    v[si * ds..(si + 1) * ds].iter().map(|a| a * inv).collect()
                })
                .collect();
            let cents = kmeans::anisotropic(
                &sub,
                &sub_dirs,
                params.k,
                params.iters,
                eta,
                params.seed ^ (si as u64) ^ 0xA1A1_A1A1,
            );
            codebooks.push(cents);
        }
        let sdc_tables = build_sdc_tables(&codebooks);
        Ok(Self {
            params,
            eta,
            codebooks,
            sdc_tables,
        })
    }
    pub fn eta(&self) -> f32 {
        self.eta
    }
}

impl Quantizer for AnisotropicPQ {
    fn dim(&self) -> usize {
        self.params.dim
    }
    fn m(&self) -> usize {
        self.params.m
    }
    fn k(&self) -> usize {
        self.params.k
    }
    fn encode(&self, x: &[f32]) -> Result<Code, PqError> {
        encode_generic(x, &self.codebooks, &self.params)
    }
    fn sdc(&self, a: &Code, b: &Code) -> f32 {
        sdc_lookup(a, b, &self.sdc_tables, self.params.k)
    }
    fn reconstruct(&self, code: &Code) -> Vec<f32> {
        reconstruct_generic(code, &self.codebooks, &self.params)
    }
}

// ============================================================================
// AnisotropicPQR — APQ preceded by a learned rotation (variance balancing).
// ============================================================================

pub struct AnisotropicPQR {
    inner: AnisotropicPQ,
    rot: rotation::Rotation,
}

impl AnisotropicPQR {
    pub fn train(train: &[Vec<f32>], params: PqParams, eta: f32) -> Result<Self, PqError> {
        params.validate()?;
        // Learn a variance-balancing orthonormal rotation (eigenbasis
        // permuted so subspaces get near-equal energy).
        let rot = rotation::Rotation::fit_variance_balancing(train, params.m, params.seed);
        let rotated: Vec<Vec<f32>> = train.iter().map(|x| rot.apply(x)).collect();
        let inner = AnisotropicPQ::train(&rotated, params, eta)?;
        Ok(Self { inner, rot })
    }
}

impl Quantizer for AnisotropicPQR {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn m(&self) -> usize {
        self.inner.m()
    }
    fn k(&self) -> usize {
        self.inner.k()
    }
    fn encode(&self, x: &[f32]) -> Result<Code, PqError> {
        let r = self.rot.apply(x);
        self.inner.encode(&r)
    }
    fn sdc(&self, a: &Code, b: &Code) -> f32 {
        self.inner.sdc(a, b)
    }
    fn reconstruct(&self, code: &Code) -> Vec<f32> {
        let r = self.inner.reconstruct(code);
        self.rot.apply_inverse(&r)
    }
}

// ============================================================================
// Shared helpers.
// ============================================================================

fn encode_generic(
    x: &[f32],
    codebooks: &[Vec<Vec<f32>>],
    p: &PqParams,
) -> Result<Code, PqError> {
    if x.len() != p.dim {
        return Err(PqError::DimMismatch {
            got: x.len(),
            expected: p.dim,
        });
    }
    let ds = p.ds();
    let mut code = vec![0u8; p.m];
    for si in 0..p.m {
        let sub = &x[si * ds..(si + 1) * ds];
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (ci, c) in codebooks[si].iter().enumerate() {
            let d = sq_l2(sub, c);
            if d < best_d {
                best_d = d;
                best = ci;
            }
        }
        code[si] = best as u8;
    }
    Ok(code)
}

fn reconstruct_generic(code: &Code, codebooks: &[Vec<Vec<f32>>], p: &PqParams) -> Vec<f32> {
    let ds = p.ds();
    let mut out = Vec::with_capacity(p.dim);
    for si in 0..p.m {
        out.extend_from_slice(&codebooks[si][code[si] as usize]);
    }
    debug_assert_eq!(out.len(), p.dim);
    let _ = ds;
    out
}

fn build_sdc_tables(codebooks: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    codebooks
        .iter()
        .map(|cb| {
            let k = cb.len();
            let mut t = vec![0.0f32; k * k];
            for i in 0..k {
                for j in i..k {
                    let d = sq_l2(&cb[i], &cb[j]);
                    t[i * k + j] = d;
                    t[j * k + i] = d;
                }
            }
            t
        })
        .collect()
}

fn sdc_lookup(a: &Code, b: &Code, tables: &[Vec<f32>], k: usize) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for si in 0..a.len() {
        let i = a[si] as usize;
        let j = b[si] as usize;
        acc += tables[si][i * k + j];
    }
    acc
}

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

// ============================================================================
// Small deterministic Gaussian data generator used by tests and the bench.
// ============================================================================

/// Generates `n` d-dim vectors from a mixture of `clusters` Gaussians.
pub fn gen_gauss(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-3.0f32..3.0f32)).collect())
        .collect();
    (0..n)
        .map(|_| {
            let c = &centers[rng.gen_range(0..clusters)];
            c.iter()
                .map(|m| m + gaussian(&mut rng) * 0.5)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn gaussian(rng: &mut StdRng) -> f32 {
    // Box-Muller.
    let u1: f32 = rng.gen_range(1e-7..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// ============================================================================
// Tests
// ============================================================================

