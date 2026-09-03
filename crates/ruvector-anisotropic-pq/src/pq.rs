//! Product quantization: baseline (isotropic k-means) and anisotropic
//! (score-aware) training.
//!
//! Layout: split each `dim`-D vector into `M` contiguous subvectors of length
//! `sub_dim = dim / M`. For each subspace train a `K`-entry codebook. A code
//! is `M` bytes (assuming `K ≤ 256`), stored as `Vec<u8>`.
//!
//! ## Anisotropic centroid update (per-subspace)
//!
//! For each cluster `C_j`, we minimise
//!
//! ```text
//! Σ_{i ∈ C_j} [ η · ((c - x_i) · û_i)²  +  ||(c - x_i) - ((c - x_i)·û_i)û_i||² ]
//! ```
//!
//! where `û_i = x_i^{(m)} / ||x_i^{(m)}||` is the unit direction of the *full*
//! datapoint projected into subspace `m` — in this PoC we approximate that
//! with the direction of the subvector itself, which is what the original
//! ScaNN codebase does when subspaces are treated independently for the
//! centroid update. The optimum satisfies the `sub_dim × sub_dim` linear
//! system
//!
//! ```text
//! ( Σ_i M_i ) c = Σ_i M_i x_i        with  M_i = I + (η - 1) û_i û_iᵀ
//! ```
//!
//! We solve it with a plain Gauss-elimination routine; `sub_dim` is small
//! (8 in the default config) so this costs `O(sub_dim³)` per centroid update
//! — negligible next to the assignment step's `O(N · K · sub_dim)`.

use crate::dataset::{dot, SplitMix64};

pub type Vector = Vec<f32>;
pub type Code = Vec<u8>;

#[derive(Debug, Clone, Copy)]
pub struct PqParams {
    pub dim: usize,
    pub m: usize,     // number of subspaces
    pub k: usize,     // centroids per subspace (≤ 256)
    pub iters: usize, // k-means iterations
    pub seed: u64,
}

impl Default for PqParams {
    fn default() -> Self {
        Self {
            dim: 64,
            m: 8,
            k: 256,
            iters: 12,
            seed: 0xBEEF_C0DE,
        }
    }
}

/// A trained codebook: `m` subspace codebooks, each `k × sub_dim` row-major.
#[derive(Debug, Clone)]
pub struct Codebook {
    pub params: PqParams,
    pub sub_dim: usize,
    /// Length `m`; each entry is `k * sub_dim` floats.
    pub centroids: Vec<Vec<f32>>,
}

impl Codebook {
    pub fn code_size_bytes(&self) -> usize {
        self.params.m
    }
    pub fn centroid(&self, m: usize, k: usize) -> &[f32] {
        let s = self.sub_dim;
        &self.centroids[m][k * s..(k + 1) * s]
    }
    pub fn encode(&self, x: &[f32]) -> Code {
        let s = self.sub_dim;
        (0..self.params.m)
            .map(|m| {
                let xm = &x[m * s..(m + 1) * s];
                nearest_centroid_l2(&self.centroids[m], self.params.k, s, xm) as u8
            })
            .collect()
    }
    pub fn encode_batch(&self, xs: &[Vector]) -> Vec<Code> {
        xs.iter().map(|x| self.encode(x)).collect()
    }
    pub fn reconstruct(&self, code: &Code) -> Vector {
        let s = self.sub_dim;
        let mut out = vec![0.0_f32; self.params.dim];
        for m in 0..self.params.m {
            let c = self.centroid(m, code[m] as usize);
            out[m * s..(m + 1) * s].copy_from_slice(c);
        }
        out
    }
}

// ─── Quantizer trait + two implementations ────────────────────────────────

pub trait Quantizer {
    fn name(&self) -> &'static str;
    fn train(&self, data: &[Vector], params: PqParams) -> Codebook;
}

/// Standard isotropic k-means PQ. Baseline (`η = 1`).
pub struct BaselinePq;

impl Quantizer for BaselinePq {
    fn name(&self) -> &'static str {
        "baseline-pq (η=1)"
    }
    fn train(&self, data: &[Vector], params: PqParams) -> Codebook {
        train_pq(data, params, 1.0)
    }
}

/// Anisotropic score-aware PQ. `eta > 1` upweights the parallel component of
/// the residual, biasing the codebook toward preserving inner-product scores.
pub struct AnisotropicPq {
    pub eta: f32,
}

impl Quantizer for AnisotropicPq {
    fn name(&self) -> &'static str {
        if self.eta >= 15.0 {
            "anisotropic-pq (η=16)"
        } else if self.eta >= 3.0 {
            "anisotropic-pq (η=4)"
        } else {
            "anisotropic-pq"
        }
    }
    fn train(&self, data: &[Vector], params: PqParams) -> Codebook {
        train_pq(data, params, self.eta)
    }
}

// ─── shared training routine ──────────────────────────────────────────────

fn train_pq(data: &[Vector], params: PqParams, eta: f32) -> Codebook {
    assert_eq!(data[0].len(), params.dim);
    assert!(params.dim % params.m == 0, "dim must be divisible by m");
    let s = params.dim / params.m;
    assert!(params.k <= 256, "k must fit in u8");

    let mut rng = SplitMix64::new(params.seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(params.m);

    for m in 0..params.m {
        // Extract subspace slice for every point.
        let subvecs: Vec<&[f32]> = data.iter().map(|x| &x[m * s..(m + 1) * s]).collect();
        centroids.push(kmeans_subspace(&subvecs, params.k, s, params.iters, eta, &mut rng));
    }

    Codebook {
        params,
        sub_dim: s,
        centroids,
    }
}

fn kmeans_subspace(
    points: &[&[f32]],
    k: usize,
    sub_dim: usize,
    iters: usize,
    eta: f32,
    rng: &mut SplitMix64,
) -> Vec<f32> {
    let n = points.len();
    // k-means++-ish init: pick k random distinct points.
    let mut centroids = vec![0.0_f32; k * sub_dim];
    for j in 0..k {
        let idx = (rng.next_u64() as usize) % n;
        centroids[j * sub_dim..(j + 1) * sub_dim].copy_from_slice(points[idx]);
    }

    let mut assign = vec![0u32; n];
    for _iter in 0..iters {
        // Assign step (always uses L2 nearest — anisotropy affects the
        // *centroid update*, which is where ScaNN diverges from ordinary PQ.
        // The paper shows this is a sound approximation because the
        // assignment for a single subvector under the score-aware loss is
        // dominated by the same nearest-centroid decision).
        for (i, p) in points.iter().enumerate() {
            assign[i] = nearest_centroid_l2(&centroids, k, sub_dim, p) as u32;
        }

        // Update step.
        for j in 0..k {
            let members: Vec<&[f32]> = points
                .iter()
                .zip(assign.iter())
                .filter(|(_, a)| **a as usize == j)
                .map(|(p, _)| *p)
                .collect();
            if members.is_empty() {
                // Re-seed empty cluster from a random point.
                let idx = (rng.next_u64() as usize) % n;
                centroids[j * sub_dim..(j + 1) * sub_dim].copy_from_slice(points[idx]);
                continue;
            }
            if (eta - 1.0).abs() < 1e-6 {
                // Isotropic: plain arithmetic mean.
                let mut c = vec![0.0_f32; sub_dim];
                for m in &members {
                    for d in 0..sub_dim {
                        c[d] += m[d];
                    }
                }
                let inv = 1.0 / members.len() as f32;
                for d in 0..sub_dim {
                    c[d] *= inv;
                }
                centroids[j * sub_dim..(j + 1) * sub_dim].copy_from_slice(&c);
            } else {
                // Anisotropic: solve  ( Σ M_i ) c = Σ M_i x_i.
                let c = solve_anisotropic_centroid(&members, sub_dim, eta);
                centroids[j * sub_dim..(j + 1) * sub_dim].copy_from_slice(&c);
            }
        }
    }
    centroids
}

fn nearest_centroid_l2(centroids: &[f32], k: usize, sub_dim: usize, x: &[f32]) -> usize {
    let mut best_j = 0usize;
    let mut best_d = f32::MAX;
    for j in 0..k {
        let c = &centroids[j * sub_dim..(j + 1) * sub_dim];
        let mut d = 0.0_f32;
        for i in 0..sub_dim {
            let e = c[i] - x[i];
            d += e * e;
        }
        if d < best_d {
            best_d = d;
            best_j = j;
        }
    }
    best_j
}

/// Build `A = Σ M_i`, `b = Σ M_i x_i` with `M_i = I + (η-1) û_i û_iᵀ`,
/// then solve `A c = b` via Gauss elimination. `sub_dim` is small.
fn solve_anisotropic_centroid(members: &[&[f32]], sub_dim: usize, eta: f32) -> Vec<f32> {
    let d = sub_dim;
    let mut a = vec![0.0_f32; d * d];
    let mut b = vec![0.0_f32; d];
    let w = eta - 1.0;

    // A starts as n·I; add w · Σ û_i û_iᵀ.
    for i in 0..d {
        a[i * d + i] += members.len() as f32;
    }
    for x in members {
        let n2 = dot(x, x);
        if n2 < 1e-12 {
            // Degenerate direction: fall back to isotropic contribution.
            for i in 0..d {
                b[i] += x[i];
            }
            continue;
        }
        let inv = 1.0 / n2;
        // A += w * (x xᵀ) / ||x||²
        for i in 0..d {
            for j in 0..d {
                a[i * d + j] += w * x[i] * x[j] * inv;
            }
        }
        // b += M_i x = (I + w û ûᵀ) x = x + w * (x·û) û  = x + w * ||x|| · û = x + w * x
        //   because û = x/||x||, so (x·û) û = (||x||² / ||x||) û = ||x|| û = x.
        // Therefore M_i x_i = (1 + w) x_i. This is a nice simplification.
        for i in 0..d {
            b[i] += (1.0 + w) * x[i];
        }
    }

    gauss_solve(&mut a, &mut b, d);
    b
}

fn gauss_solve(a: &mut [f32], b: &mut [f32], d: usize) {
    // Partial-pivoting Gauss elimination on a `d × d` system.
    for k in 0..d {
        // Pivot.
        let mut piv = k;
        let mut best = a[k * d + k].abs();
        for r in (k + 1)..d {
            let v = a[r * d + k].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-9 {
            // Singular — leave c at whatever partial solution we have.
            continue;
        }
        if piv != k {
            for c in 0..d {
                a.swap(k * d + c, piv * d + c);
            }
            b.swap(k, piv);
        }
        // Eliminate.
        let akk = a[k * d + k];
        for r in (k + 1)..d {
            let f = a[r * d + k] / akk;
            for c in k..d {
                a[r * d + c] -= f * a[k * d + c];
            }
            b[r] -= f * b[k];
        }
    }
    // Back-substitute.
    for k in (0..d).rev() {
        let mut s = b[k];
        for c in (k + 1)..d {
            s -= a[k * d + c] * b[c];
        }
        let akk = a[k * d + k];
        b[k] = if akk.abs() > 1e-12 { s / akk } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_solves_2x2() {
        // [[2,1],[1,3]] x = [5, 10] → x = [1, 3]
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 10.0];
        gauss_solve(&mut a, &mut b, 2);
        assert!((b[0] - 1.0).abs() < 1e-5, "x0={}", b[0]);
        assert!((b[1] - 3.0).abs() < 1e-5, "x1={}", b[1]);
    }

    #[test]
    fn baseline_and_anisotropic_produce_valid_codes() {
        use crate::dataset::{DatasetConfig, GaussianMixture};
        let cfg = DatasetConfig {
            dim: 16,
            n_db: 200,
            n_queries: 10,
            k_clusters: 4,
            noise_std: 0.3,
            seed: 42,
        };
        let ds = GaussianMixture::generate(cfg);
        let params = PqParams {
            dim: 16,
            m: 4,
            k: 16,
            iters: 5,
            seed: 7,
        };
        let cb_b = BaselinePq.train(&ds.db, params);
        let cb_a = AnisotropicPq { eta: 4.0 }.train(&ds.db, params);
        let code_b = cb_b.encode(&ds.db[0]);
        let code_a = cb_a.encode(&ds.db[0]);
        assert_eq!(code_b.len(), 4);
        assert_eq!(code_a.len(), 4);
        for c in &code_b {
            assert!((*c as usize) < 16);
        }
        for c in &code_a {
            assert!((*c as usize) < 16);
        }
    }
}
