//! Product Quantizer with pluggable training loss.
//!
//! - QuantizerKind::Mse:        ordinary Lloyd k-means (isotropic baseline).
//! - QuantizerKind::Anisotropic { eta }: loss-aware k-means where the
//!   assignment cost is `eta * r_parallel^2 + r_perpendicular^2` measured
//!   against the *full* training vector (not just the subvector), so the
//!   parallel direction is meaningful.
//!
//! The training data is partitioned into M equal-sized subvectors of length
//! D/M, each with its own 256-entry codebook (8 bits per subquantizer).

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::loss::decompose_residual;

#[derive(Clone, Copy, Debug)]
pub enum QuantizerKind {
    Mse,
    Anisotropic { eta: f32 },
}

pub struct ProductQuantizer {
    pub d: usize,
    pub m: usize,         // number of subquantizers
    pub k: usize,         // codebook size (typically 256)
    pub kind: QuantizerKind,
    pub codebooks: Vec<Vec<f32>>, // codebooks[m] flat [k * (d/m)]
}

impl ProductQuantizer {
    pub fn ds(&self) -> usize {
        self.d / self.m
    }

    /// Train PQ codebooks on `data` (N rows of length D).
    pub fn train(
        d: usize,
        m: usize,
        k: usize,
        kind: QuantizerKind,
        data: &[Vec<f32>],
        iters: usize,
        seed: u64,
    ) -> Self {
        assert!(d % m == 0, "D must divide M");
        assert!(k <= data.len(), "need at least k training points");
        let ds = d / m;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut codebooks = Vec::with_capacity(m);

        for sub in 0..m {
            let start = sub * ds;
            let end = start + ds;
            let book = train_subquantizer(data, start, end, k, kind, iters, &mut rng);
            codebooks.push(book);
        }

        Self { d, m, k, kind, codebooks }
    }

    /// Encode one vector into `m` codeword indices.
    ///
    /// Following ScaNN, encoding is per-subspace nearest-centroid (Euclidean).
    /// The anisotropic loss affects *codebook training*; given trained
    /// codebooks, the optimal assignment for a vector is independent per
    /// subspace because the loss decomposes additively across subspaces for
    /// unit-norm full vectors.
    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        let ds = self.ds();
        let mut codes = vec![0u8; self.m];
        for sub in 0..self.m {
            let start = sub * ds;
            let book = &self.codebooks[sub];
            let mut best = 0usize;
            let mut best_cost = f32::INFINITY;
            for c in 0..self.k {
                let mut s = 0.0f32;
                for j in 0..ds {
                    let d = x[start + j] - book[c * ds + j];
                    s += d * d;
                }
                if s < best_cost {
                    best_cost = s;
                    best = c;
                }
            }
            codes[sub] = best as u8;
        }
        codes
    }

    /// Reconstruct vector from codes.
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let ds = self.ds();
        let mut x = vec![0.0f32; self.d];
        for sub in 0..self.m {
            let c = codes[sub] as usize;
            let book = &self.codebooks[sub];
            for j in 0..ds {
                x[sub * ds + j] = book[c * ds + j];
            }
        }
        x
    }

    /// Compute the symmetric inner-product distance lookup table for a query:
    ///   lut[sub][c] = <q_sub, codebook[sub][c]>
    /// Used at search time to score codes in O(M) instead of O(D).
    pub fn build_ip_lut(&self, q: &[f32]) -> Vec<Vec<f32>> {
        let ds = self.ds();
        let mut lut = Vec::with_capacity(self.m);
        for sub in 0..self.m {
            let start = sub * ds;
            let book = &self.codebooks[sub];
            let mut row = vec![0.0f32; self.k];
            for c in 0..self.k {
                let mut s = 0.0f32;
                for j in 0..ds {
                    s += q[start + j] * book[c * ds + j];
                }
                row[c] = s;
            }
            lut.push(row);
        }
        lut
    }

    /// Score one code against a precomputed inner-product LUT.
    /// Higher = more similar (for MIPS / cosine on unit-norm vectors).
    pub fn score_code(&self, lut: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut s = 0.0f32;
        for sub in 0..self.m {
            s += lut[sub][codes[sub] as usize];
        }
        s
    }
}

fn train_subquantizer(
    data: &[Vec<f32>],
    start: usize,
    end: usize,
    k: usize,
    kind: QuantizerKind,
    iters: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    let ds = end - start;
    let n = data.len();

    // k-means++ seeding on subvectors (always Euclidean — anisotropic
    // weighting kicks in during assignment).
    let mut centroids = Vec::with_capacity(k * ds);
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&data[first][start..end]);

    let mut d2 = vec![f32::INFINITY; n];
    for _ in 1..k {
        let c_off = centroids.len() - ds;
        for i in 0..n {
            let mut s = 0.0f32;
            for j in 0..ds {
                let d = data[i][start + j] - centroids[c_off + j];
                s += d * d;
            }
            if s < d2[i] {
                d2[i] = s;
            }
        }
        let total: f32 = d2.iter().sum();
        let mut t = rng.gen::<f32>() * total;
        let mut pick = n - 1;
        for i in 0..n {
            t -= d2[i];
            if t <= 0.0 {
                pick = i;
                break;
            }
        }
        centroids.extend_from_slice(&data[pick][start..end]);
    }

    // Iterate. Assignment uses the chosen loss; update is the mean of
    // assigned subvectors (closed form for L2; an approximation for the
    // anisotropic loss that still monotonically decreases total cost in
    // practice — same trick ScaNN uses to keep training cheap).
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        // assignment
        for i in 0..n {
            let mut best = 0u32;
            let mut best_cost = f32::INFINITY;
            for c in 0..k {
                let c_off = c * ds;
                let cost = match kind {
                    QuantizerKind::Mse => {
                        let mut s = 0.0f32;
                        for j in 0..ds {
                            let d = data[i][start + j] - centroids[c_off + j];
                            s += d * d;
                        }
                        s
                    }
                    QuantizerKind::Anisotropic { eta } => {
                        // Decompose against the full data vector; only this
                        // subspace's residual contributes, but the parallel
                        // direction is the full x — that's the whole point.
                        let x = &data[i];
                        let mut rdotx = 0.0f32;
                        let mut rnorm2 = 0.0f32;
                        let mut xnorm2 = 0.0f32;
                        for j in 0..x.len() {
                            let xi = x[j];
                            xnorm2 += xi * xi;
                            // residual is 0 outside this subspace under
                            // single-subspace replacement.
                            let ri = if j >= start && j < end {
                                xi - centroids[c_off + (j - start)]
                            } else {
                                0.0
                            };
                            rdotx += ri * xi;
                            rnorm2 += ri * ri;
                        }
                        let par = if xnorm2 > f32::EPSILON {
                            (rdotx * rdotx) / xnorm2
                        } else {
                            0.0
                        };
                        let perp = (rnorm2 - par).max(0.0);
                        eta * par + perp
                    }
                };
                if cost < best_cost {
                    best_cost = cost;
                    best = c as u32;
                }
            }
            assign[i] = best;
        }

        // Update step.
        //
        // For QuantizerKind::Mse we use the closed-form mean.
        //
        // For QuantizerKind::Anisotropic { eta } we use the closed-form
        // anisotropic centroid update derived in Guo et al. (ICML 2020),
        // specialized to unit-norm full vectors and single-subspace PQ
        // replacement. With y_i the i-th subvector and the parallel
        // direction equal to the full vector x_i (unit-norm), the loss
        // becomes
        //
        //   L_i(c) = ||y_i - c||^2 + (eta - 1) * (<y_i - c, y_i>)^2.
        //
        // Setting the gradient to zero gives the per-cluster linear system
        //   ( |S| I + (eta-1) sum_i y_i y_i^T ) c = sum_i y_i + (eta-1) sum_i ||y_i||^2 y_i.
        //
        // ds is small (typ. 4..16) so we solve via Gaussian elimination.
        match kind {
            QuantizerKind::Mse => {
                let mut sums = vec![0.0f32; k * ds];
                let mut counts = vec![0u32; k];
                for i in 0..n {
                    let a = assign[i] as usize;
                    counts[a] += 1;
                    for j in 0..ds {
                        sums[a * ds + j] += data[i][start + j];
                    }
                }
                for c in 0..k {
                    if counts[c] == 0 {
                        let r = rng.gen_range(0..n);
                        for j in 0..ds {
                            centroids[c * ds + j] = data[r][start + j];
                        }
                    } else {
                        let inv = 1.0 / counts[c] as f32;
                        for j in 0..ds {
                            centroids[c * ds + j] = sums[c * ds + j] * inv;
                        }
                    }
                }
            }
            QuantizerKind::Anisotropic { eta } => {
                let scale = (eta - 1.0).max(0.0);
                // A_S = |S| I + scale * sum y_i y_i^T   (ds x ds)
                // b_S = sum y_i + scale * sum ||y_i||^2 y_i
                let mut a_mats = vec![0.0f32; k * ds * ds];
                let mut b_vecs = vec![0.0f32; k * ds];
                let mut counts = vec![0u32; k];
                for i in 0..n {
                    let a = assign[i] as usize;
                    counts[a] += 1;
                    let y = &data[i][start..end];
                    let yn2: f32 = y.iter().map(|v| v * v).sum();
                    let a_off = a * ds * ds;
                    let b_off = a * ds;
                    for r in 0..ds {
                        b_vecs[b_off + r] += y[r] + scale * yn2 * y[r];
                        for c in 0..ds {
                            a_mats[a_off + r * ds + c] += scale * y[r] * y[c];
                        }
                    }
                }
                for c in 0..k {
                    if counts[c] == 0 {
                        let r = rng.gen_range(0..n);
                        for j in 0..ds {
                            centroids[c * ds + j] = data[r][start + j];
                        }
                        continue;
                    }
                    // Add |S| * I to the rank-1 sum to form A_S.
                    let nc = counts[c] as f32;
                    let a_off = c * ds * ds;
                    for r in 0..ds {
                        a_mats[a_off + r * ds + r] += nc;
                    }
                    let mut a_local: Vec<f32> = a_mats[a_off..a_off + ds * ds].to_vec();
                    let mut b_local: Vec<f32> = b_vecs[c * ds..c * ds + ds].to_vec();
                    if solve_in_place(&mut a_local, &mut b_local, ds) {
                        for j in 0..ds {
                            centroids[c * ds + j] = b_local[j];
                        }
                    } else {
                        // Degenerate system — fall back to plain mean.
                        let mut s = vec![0.0f32; ds];
                        for i in 0..n {
                            if assign[i] as usize == c {
                                for j in 0..ds {
                                    s[j] += data[i][start + j];
                                }
                            }
                        }
                        let inv = 1.0 / nc;
                        for j in 0..ds {
                            centroids[c * ds + j] = s[j] * inv;
                        }
                    }
                }
            }
        }
    }

    centroids
}

/// In-place Gaussian elimination with partial pivoting. Solves `a * x = b`
/// for `x`, writing the result into `b`. Returns false if the matrix is
/// numerically singular.
fn solve_in_place(a: &mut [f32], b: &mut [f32], n: usize) -> bool {
    for i in 0..n {
        // pivot
        let mut piv = i;
        let mut piv_val = a[i * n + i].abs();
        for r in (i + 1)..n {
            let v = a[r * n + i].abs();
            if v > piv_val {
                piv = r;
                piv_val = v;
            }
        }
        if piv_val < 1e-10 {
            return false;
        }
        if piv != i {
            for c in 0..n {
                a.swap(i * n + c, piv * n + c);
            }
            b.swap(i, piv);
        }
        let inv = 1.0 / a[i * n + i];
        for r in (i + 1)..n {
            let f = a[r * n + i] * inv;
            if f == 0.0 {
                continue;
            }
            for c in i..n {
                a[r * n + c] -= f * a[i * n + c];
            }
            b[r] -= f * b[i];
        }
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for c in (i + 1)..n {
            s -= a[i * n + c] * b[c];
        }
        b[i] = s / a[i * n + i];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn unit_norm(v: &mut [f32]) {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            for x in v.iter_mut() {
                *x /= n;
            }
        }
    }

    fn random_unit_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
                unit_norm(&mut v);
                v
            })
            .collect()
    }

    #[test]
    fn mse_pq_trains_and_reconstructs() {
        let data = random_unit_vectors(256, 16, 1);
        let pq = ProductQuantizer::train(16, 4, 16, QuantizerKind::Mse, &data, 8, 42);
        let codes = pq.encode(&data[0]);
        let recon = pq.decode(&codes);
        assert_eq!(recon.len(), 16);
        let err: f32 = data[0]
            .iter()
            .zip(&recon)
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(err < 1.0, "reconstruction error too large: {err}");
    }

    #[test]
    fn anisotropic_reduces_parallel_residual_energy() {
        // Direct, deterministic property: AVQ-trained codebooks should
        // produce smaller mean parallel residual energy than MSE-trained
        // codebooks — that's the *whole point* of the anisotropic loss.
        // (Top-1 ranking on small data is too noisy to assert.)
        let data = random_unit_vectors(512, 16, 7);
        let pq_mse = ProductQuantizer::train(16, 4, 16, QuantizerKind::Mse, &data, 15, 1);
        let pq_aniso =
            ProductQuantizer::train(16, 4, 16, QuantizerKind::Anisotropic { eta: 4.0 }, &data, 15, 1);

        let mse_par = mean_parallel_energy(&pq_mse, &data);
        let aniso_par = mean_parallel_energy(&pq_aniso, &data);
        assert!(
            aniso_par < mse_par,
            "aniso parallel energy {aniso_par} must be < mse parallel energy {mse_par}"
        );
    }

    fn mean_parallel_energy(pq: &ProductQuantizer, data: &[Vec<f32>]) -> f32 {
        let mut total = 0.0f32;
        for x in data {
            let recon = pq.decode(&pq.encode(x));
            let (par, _perp) = crate::loss::decompose_residual(x, &recon);
            total += par;
        }
        total / data.len() as f32
    }
}
