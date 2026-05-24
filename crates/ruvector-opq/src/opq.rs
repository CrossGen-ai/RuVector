//! Optimized Product Quantization (Ge, He, Ke, Sun, CVPR 2013).
//!
//! Both variants learn a `d × d` orthogonal rotation `R` such that PQ on the
//! rotated data `y = R x` has lower reconstruction error than PQ on `x`
//! directly.  Decoded reconstruction is `R^T ŷ`, so the rotation can be
//! treated as a free invertible reparameterisation.
//!
//! * `OpqNp` — closed-form eigenvalue allocation: pick R by sorting principal
//!   components into `m` buckets with balanced product of variances.  Costs
//!   **one PCA**, no PQ iterations.  Good baseline; typically halves PQ MSE.
//! * `OpqP`  — iterative parametric refinement via orthogonal Procrustes:
//!   alternates (a) train PQ on `R X`, (b) re-solve `R = V U^T` from SVD of
//!   `X Ŷ^T`.  Costs `iters × PQ-train + iters × SVD(d)`; lower MSE than NP.

use crate::pq::Pq;
use crate::Quantizer;
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct OpqNp {
    pub d: usize,
    pub m: usize,
    pub ds: usize,
    /// `d × d` row-major rotation matrix.  `y = R x`.
    pub rotation: Vec<f32>,
    /// PQ trained on rotated data.
    pub pq: Pq,
    /// Mean of training data; rotation acts after centring.
    pub mean: Vec<f32>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OpqP {
    pub d: usize,
    pub m: usize,
    pub ds: usize,
    pub rotation: Vec<f32>,
    pub pq: Pq,
    pub mean: Vec<f32>,
    pub iters: usize,
}

impl OpqNp {
    pub fn new(d: usize, m: usize) -> Self {
        assert!(d % m == 0);
        Self {
            d,
            m,
            ds: d / m,
            rotation: identity(d),
            pq: Pq::new(d, m),
            mean: vec![0.0; d],
        }
    }
}

impl OpqP {
    pub fn new(d: usize, m: usize, iters: usize) -> Self {
        assert!(d % m == 0);
        Self {
            d,
            m,
            ds: d / m,
            rotation: identity(d),
            pq: Pq::new(d, m),
            mean: vec![0.0; d],
            iters,
        }
    }
}

fn identity(d: usize) -> Vec<f32> {
    let mut r = vec![0.0f32; d * d];
    for i in 0..d {
        r[i * d + i] = 1.0;
    }
    r
}

/// Apply `y = R (x - mean)` for one vector.
fn apply_rotation(rot: &[f32], mean: &[f32], x: &[f32], y: &mut [f32]) {
    let d = mean.len();
    for i in 0..d {
        let mut s = 0.0f32;
        for j in 0..d {
            s += rot[i * d + j] * (x[j] - mean[j]);
        }
        y[i] = s;
    }
}

/// Apply `x = R^T y + mean` (decode rotation).
fn apply_rotation_t(rot: &[f32], mean: &[f32], y: &[f32], x: &mut [f32]) {
    let d = mean.len();
    for j in 0..d {
        let mut s = 0.0f32;
        for i in 0..d {
            s += rot[i * d + j] * y[i];
        }
        x[j] = s + mean[j];
    }
}

/// Compute mean (`d`) and rotate every row through `R` in-place,
/// returning a new buffer of rotated, centred vectors.
fn rotate_dataset(rot: &[f32], mean: &[f32], data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        apply_rotation(rot, mean, &data[i * d..(i + 1) * d], &mut out[i * d..(i + 1) * d]);
    }
    out
}

fn mean_of(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; d];
    for i in 0..n {
        for j in 0..d {
            m[j] += data[i * d + j];
        }
    }
    let inv = 1.0 / n as f32;
    for j in 0..d {
        m[j] *= inv;
    }
    m
}

/// Eigenvalue-allocation rotation (Ge §4.1, non-parametric).
///
/// 1. centre data, build `Σ = X^T X / n` (d×d covariance),
/// 2. eigendecompose Σ → variances σ²_i + principal axes u_i,
/// 3. balanced greedy assignment of axes to `m` buckets so each bucket has
///    roughly equal product of variances (= equal sum of log-variances),
/// 4. rotation rows = bucket-ordered eigenvectors.
fn eigen_allocation_rotation(data: &[f32], mean: &[f32], n: usize, d: usize, m: usize) -> Vec<f32> {
    // Build covariance.
    let mut cov = DMatrix::<f32>::zeros(d, d);
    for i in 0..n {
        for a in 0..d {
            let va = data[i * d + a] - mean[a];
            for b in 0..d {
                let vb = data[i * d + b] - mean[b];
                cov[(a, b)] += va * vb;
            }
        }
    }
    cov /= n as f32;

    let eig = SymmetricEigen::new(cov);
    // eigenvalues: vector length d (unordered); eigenvectors: d×d columns.
    let vals = eig.eigenvalues;
    let vecs = eig.eigenvectors;

    // Sort indices by eigenvalue descending.
    let mut idx: Vec<usize> = (0..d).collect();
    idx.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Balanced greedy: keep `m` bucket log-variance sums; assign next largest
    // eigenvalue to the bucket with the smallest sum.
    let ds = d / m;
    let mut buckets: Vec<Vec<usize>> = vec![Vec::with_capacity(ds); m];
    let mut sums = vec![0.0f64; m];
    let cap = ds;
    for &i in &idx {
        let v = (vals[i].max(1e-12) as f64).ln();
        // Pick smallest-sum bucket that still has room.
        let mut pick = 0usize;
        let mut best = f64::INFINITY;
        for b in 0..m {
            if buckets[b].len() < cap && sums[b] < best {
                best = sums[b];
                pick = b;
            }
        }
        buckets[pick].push(i);
        sums[pick] += v;
    }

    // Build R (row-major d×d).  Rows = stacked bucket eigenvectors.
    let mut rot = vec![0.0f32; d * d];
    let mut row = 0usize;
    for b in 0..m {
        for &i in &buckets[b] {
            for j in 0..d {
                rot[row * d + j] = vecs[(j, i)];
            }
            row += 1;
        }
    }
    debug_assert_eq!(row, d);
    rot
}

/// Orthogonal Procrustes: given `X` (d×n) and `Ŷ` (d×n), return `R` (d×d)
/// minimising `‖R X − Ŷ‖_F`. Closed-form: SVD(X Ŷ^T) = U Σ V^T → R = V U^T.
fn procrustes(x: &DMatrix<f32>, yhat: &DMatrix<f32>) -> DMatrix<f32> {
    let m = x * yhat.transpose(); // d × d
    let svd = m.svd(true, true);
    let u = svd.u.expect("u");
    let vt = svd.v_t.expect("v_t");
    // R = V U^T  =  vt^T * u^T
    vt.transpose() * u.transpose()
}

fn dmat_to_rot(m: &DMatrix<f32>) -> Vec<f32> {
    let d = m.nrows();
    let mut r = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            r[i * d + j] = m[(i, j)];
        }
    }
    r
}

impl Quantizer for OpqNp {
    fn fit(&mut self, data: &[f32], n: usize, d: usize) {
        assert_eq!(d, self.d);
        self.mean = mean_of(data, n, d);
        self.rotation = eigen_allocation_rotation(data, &self.mean, n, d, self.m);
        let rotated = rotate_dataset(&self.rotation, &self.mean, data, n, d);
        self.pq.fit(&rotated, n, d);
    }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        let mut y = vec![0.0f32; self.d];
        apply_rotation(&self.rotation, &self.mean, x, &mut y);
        self.pq.encode(&y, out);
    }
    fn decode(&self, code: &[u8], out: &mut [f32]) {
        let mut y = vec![0.0f32; self.d];
        self.pq.decode(code, &mut y);
        apply_rotation_t(&self.rotation, &self.mean, &y, out);
    }
    fn adc(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut y = vec![0.0f32; self.d];
        apply_rotation(&self.rotation, &self.mean, query, &mut y);
        self.pq.adc(&y, code)
    }
    fn build_lut(&self, query: &[f32], lut: &mut [f32]) {
        let mut y = vec![0.0f32; self.d];
        apply_rotation(&self.rotation, &self.mean, query, &mut y);
        self.pq.build_lut(&y, lut);
    }
    fn m(&self) -> usize {
        self.m
    }
    fn d(&self) -> usize {
        self.d
    }
}

impl Quantizer for OpqP {
    fn fit(&mut self, data: &[f32], n: usize, d: usize) {
        assert_eq!(d, self.d);
        self.mean = mean_of(data, n, d);

        // Warm start: OPQ-NP rotation.
        self.rotation = eigen_allocation_rotation(data, &self.mean, n, d, self.m);

        // Centred data X as (d × n) column-major DMatrix.
        let mut xmat = DMatrix::<f32>::zeros(d, n);
        for i in 0..n {
            for j in 0..d {
                xmat[(j, i)] = data[i * d + j] - self.mean[j];
            }
        }

        for _ in 0..self.iters {
            // Step A: rotate data, retrain PQ on rotated data.
            let rotated = rotate_dataset(&self.rotation, &self.mean, data, n, d);
            self.pq = Pq::new(d, self.m);
            self.pq.fit(&rotated, n, d);

            // Step B: build reconstructions Ŷ_i  (in rotated space).
            let mut yhat = DMatrix::<f32>::zeros(d, n);
            let mut code = vec![0u8; self.m];
            let mut rec = vec![0.0f32; d];
            for i in 0..n {
                let yslice = &rotated[i * d..(i + 1) * d];
                self.pq.encode(yslice, &mut code);
                self.pq.decode(&code, &mut rec);
                for j in 0..d {
                    yhat[(j, i)] = rec[j];
                }
            }

            // Step C: Procrustes update — R = V U^T from SVD(X Ŷ^T).
            let r_new = procrustes(&xmat, &yhat);
            self.rotation = dmat_to_rot(&r_new);
        }

        // Final PQ retrain at the converged rotation.
        let rotated = rotate_dataset(&self.rotation, &self.mean, data, n, d);
        self.pq = Pq::new(d, self.m);
        self.pq.fit(&rotated, n, d);
    }
    fn encode(&self, x: &[f32], out: &mut [u8]) {
        let mut y = vec![0.0f32; self.d];
        apply_rotation(&self.rotation, &self.mean, x, &mut y);
        self.pq.encode(&y, out);
    }
    fn decode(&self, code: &[u8], out: &mut [f32]) {
        let mut y = vec![0.0f32; self.d];
        self.pq.decode(code, &mut y);
        apply_rotation_t(&self.rotation, &self.mean, &y, out);
    }
    fn adc(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut y = vec![0.0f32; self.d];
        apply_rotation(&self.rotation, &self.mean, query, &mut y);
        self.pq.adc(&y, code)
    }
    fn build_lut(&self, query: &[f32], lut: &mut [f32]) {
        let mut y = vec![0.0f32; self.d];
        apply_rotation(&self.rotation, &self.mean, query, &mut y);
        self.pq.build_lut(&y, lut);
    }
    fn m(&self) -> usize {
        self.m
    }
    fn d(&self) -> usize {
        self.d
    }
}

/// Quick orthogonality + size check on a stored rotation.
pub fn is_orthogonal(rot: &[f32], d: usize, tol: f32) -> bool {
    // Check ‖R R^T − I‖_F < tol.
    let mut err = 0.0f32;
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0f32;
            for k in 0..d {
                s += rot[i * d + k] * rot[j * d + k];
            }
            let target = if i == j { 1.0 } else { 0.0 };
            let e = s - target;
            err += e * e;
        }
    }
    err.sqrt() < tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn anisotropic(n: usize, d: usize, seed: u64) -> Vec<f32> {
        // Highly anisotropic Gaussian: variance decays by axis.
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                let scale = ((d - j) as f32).sqrt();
                // Box-Muller-ish: sum of 4 uniforms approximates Gaussian.
                let u: f32 = (0..4).map(|_| rng.gen::<f32>() - 0.5).sum();
                data[i * d + j] = u * scale;
            }
        }
        data
    }

    #[test]
    fn opq_np_rotation_is_orthogonal() {
        let d = 16;
        let m = 4;
        let n = 800;
        let data = anisotropic(n, d, 11);
        let mut q = OpqNp::new(d, m);
        q.fit(&data, n, d);
        assert!(is_orthogonal(&q.rotation, d, 1e-3));
    }

    #[test]
    fn opq_np_beats_pq_on_anisotropic_data() {
        let d = 32;
        let m = 8;
        let n = 1500;
        let data = anisotropic(n, d, 7);

        let mut pq = Pq::new(d, m);
        pq.fit(&data, n, d);
        let mut opq = OpqNp::new(d, m);
        opq.fit(&data, n, d);

        let mut code = vec![0u8; m];
        let mut rec = vec![0.0f32; d];
        let mut pq_err = 0.0f64;
        let mut opq_err = 0.0f64;
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            pq.encode(x, &mut code);
            pq.decode(&code, &mut rec);
            for j in 0..d {
                let e = (x[j] - rec[j]) as f64;
                pq_err += e * e;
            }
            opq.encode(x, &mut code);
            opq.decode(&code, &mut rec);
            for j in 0..d {
                let e = (x[j] - rec[j]) as f64;
                opq_err += e * e;
            }
        }
        assert!(
            opq_err < pq_err,
            "OPQ-NP must beat PQ on anisotropic data: pq={} opq={}",
            pq_err,
            opq_err
        );
    }

    #[test]
    fn opq_p_beats_opq_np_with_enough_iters() {
        let d = 24;
        let m = 6;
        let n = 1500;
        let data = anisotropic(n, d, 21);

        let mut np = OpqNp::new(d, m);
        np.fit(&data, n, d);
        let mut p = OpqP::new(d, m, 4);
        p.fit(&data, n, d);

        let mut code = vec![0u8; m];
        let mut rec = vec![0.0f32; d];
        let mut np_err = 0.0f64;
        let mut p_err = 0.0f64;
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            np.encode(x, &mut code);
            np.decode(&code, &mut rec);
            for j in 0..d {
                let e = (x[j] - rec[j]) as f64;
                np_err += e * e;
            }
            p.encode(x, &mut code);
            p.decode(&code, &mut rec);
            for j in 0..d {
                let e = (x[j] - rec[j]) as f64;
                p_err += e * e;
            }
        }
        // Parametric should never be worse than NP after a few iterations;
        // we allow a small slack for k-means stochasticity.
        assert!(p_err <= np_err * 1.05, "p={} np={}", p_err, np_err);
    }
}
