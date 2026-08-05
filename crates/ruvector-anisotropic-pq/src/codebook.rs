//! Codebook training: standard (isotropic) Lloyd's k-means and the
//! score-aware anisotropic variant from ScaNN (Guo et al., ICML 2020).
//!
//! Both trainers implement [`PqCodebookTrainer`] so the index side can pick
//! one at runtime and downstream benches can plot them on the same axes.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::{l2_sq, norm_sq, PqError};

/// Training hyper-parameters shared by both trainers.
#[derive(Debug, Clone, Copy)]
pub struct PqTrainConfig {
    /// Number of sub-spaces (must divide the vector dim).
    pub m: usize,
    /// Codebook size per sub-space. 256 → 8-bit codes.
    pub k: usize,
    /// Maximum Lloyd iterations.
    pub iters: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for PqTrainConfig {
    fn default() -> Self {
        Self { m: 8, k: 256, iters: 20, seed: 0xA150_7C0Bu64 }
    }
}

/// A trained PQ codebook: `centroids[sub][code]` is a `d = D/M`-dim vector.
#[derive(Debug, Clone)]
pub struct PqCodebook {
    /// Number of sub-spaces.
    pub m: usize,
    /// Codebook size per sub-space.
    pub k: usize,
    /// Sub-vector dimension (`D/M`).
    pub d: usize,
    /// Flat storage: `centroids[sub*k*d + code*d + j]`.
    pub centroids: Vec<f32>,
}

impl PqCodebook {
    /// Return the `d`-dim centroid slice for `(sub, code)`.
    #[inline]
    pub fn centroid(&self, sub: usize, code: usize) -> &[f32] {
        let start = sub * self.k * self.d + code * self.d;
        &self.centroids[start..start + self.d]
    }

    /// Encode a full vector to `M` byte codes (K must be ≤ 256).
    pub fn encode(&self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(out.len(), self.m);
        debug_assert_eq!(x.len(), self.m * self.d);
        for sub in 0..self.m {
            let sv = &x[sub * self.d..(sub + 1) * self.d];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..self.k {
                let dist = l2_sq(sv, self.centroid(sub, c));
                if dist < best_d {
                    best_d = dist;
                    best = c;
                }
            }
            out[sub] = best as u8;
        }
    }

    /// Estimated heap bytes.
    pub fn memory_bytes(&self) -> usize {
        self.centroids.len() * std::mem::size_of::<f32>()
    }
}

/// A codebook trainer. Isotropic and anisotropic variants both implement this.
pub trait PqCodebookTrainer {
    /// Train a codebook from `train` (row-major, N × D).
    fn train(&self, train: &[Vec<f32>], cfg: &PqTrainConfig) -> Result<PqCodebook, PqError>;
    /// Human-readable name for reporting.
    fn name(&self) -> &'static str;
}

fn split_shape(dim: usize, m: usize) -> Result<usize, PqError> {
    if dim == 0 || m == 0 || dim % m != 0 {
        return Err(PqError::BadShape { dim, m });
    }
    Ok(dim / m)
}

fn kmeans_pp_init(sv: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    // Classic k-means++ seeding on the sub-space slice.
    let n = sv.len();
    let mut idxs: Vec<usize> = (0..n).collect();
    idxs.shuffle(rng);
    let first = idxs[0];
    let mut centers: Vec<Vec<f32>> = vec![sv[first].clone()];
    let mut d2: Vec<f32> = sv.iter().map(|x| l2_sq(x, &centers[0])).collect();
    while centers.len() < k {
        let total: f32 = d2.iter().sum();
        if total <= 0.0 {
            // Degenerate — fall back to any distinct point.
            let idx = centers.len() % n;
            centers.push(sv[idx].clone());
        } else {
            let mut acc = 0.0f32;
            let target = rand::Rng::gen_range(rng, 0.0..total);
            let mut chosen = n - 1;
            for (i, &d) in d2.iter().enumerate() {
                acc += d;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            centers.push(sv[chosen].clone());
        }
        let last = centers.last().unwrap();
        for (i, x) in sv.iter().enumerate() {
            let d = l2_sq(x, last);
            if d < d2[i] {
                d2[i] = d;
            }
        }
    }
    centers
}

/// Standard isotropic Lloyd's k-means per sub-space.
#[derive(Debug, Default, Clone, Copy)]
pub struct IsotropicTrainer;

impl PqCodebookTrainer for IsotropicTrainer {
    fn name(&self) -> &'static str {
        "isotropic-pq"
    }

    fn train(&self, train: &[Vec<f32>], cfg: &PqTrainConfig) -> Result<PqCodebook, PqError> {
        let n = train.len();
        if n < cfg.k {
            return Err(PqError::TooFewSamples { n, k: cfg.k });
        }
        let dim = train[0].len();
        let d = split_shape(dim, cfg.m)?;
        let mut centroids = vec![0.0f32; cfg.m * cfg.k * d];
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        for sub in 0..cfg.m {
            let sv: Vec<Vec<f32>> = train
                .iter()
                .map(|x| x[sub * d..(sub + 1) * d].to_vec())
                .collect();
            let mut centers = kmeans_pp_init(&sv, cfg.k, &mut rng);
            for _ in 0..cfg.iters {
                let mut sums = vec![vec![0.0f32; d]; cfg.k];
                let mut counts = vec![0u32; cfg.k];
                for x in &sv {
                    let mut best = 0usize;
                    let mut best_d = f32::INFINITY;
                    for (c, cv) in centers.iter().enumerate() {
                        let dd = l2_sq(x, cv);
                        if dd < best_d {
                            best_d = dd;
                            best = c;
                        }
                    }
                    counts[best] += 1;
                    let s = &mut sums[best];
                    for j in 0..d {
                        s[j] += x[j];
                    }
                }
                for c in 0..cfg.k {
                    if counts[c] > 0 {
                        let inv = 1.0 / counts[c] as f32;
                        for j in 0..d {
                            centers[c][j] = sums[c][j] * inv;
                        }
                    }
                }
            }
            let base = sub * cfg.k * d;
            for c in 0..cfg.k {
                centroids[base + c * d..base + (c + 1) * d].copy_from_slice(&centers[c]);
            }
        }
        Ok(PqCodebook { m: cfg.m, k: cfg.k, d, centroids })
    }
}

/// Score-aware anisotropic trainer.
///
/// The per-point weighted loss (following ScaNN) is
///
/// ```text
/// L(x, c) = h_par * ||r_∥||² + h_orth * ||r_⊥||²
///         where r = x - c, u = x / ||x||,
///               r_∥ = (r · u) u, r_⊥ = r - r_∥
/// ```
///
/// With `η = h_par / h_orth ≥ 1`, this reduces to standard k-means at `η = 1`
/// and to a rank-preserving loss for MIPS as `η → ∞`. The optimal centroid
/// per Lloyd step is the solution of
///
/// ```text
/// (Σᵢ Mᵢ) c* = Σᵢ Mᵢ xᵢ ,  Mᵢ = h_orth I + (h_par - h_orth) uᵢuᵢᵀ
/// ```
///
/// which is a tiny `d × d` symmetric positive-definite system.
#[derive(Debug, Clone, Copy)]
pub struct AnisotropicTrainer {
    /// Weight ratio η = h_par / h_orth (≥ 1). Common values: 2, 4, 8, 16.
    pub eta: f32,
}

impl AnisotropicTrainer {
    /// Construct with a given `η`.
    pub fn new(eta: f32) -> Self { Self { eta: eta.max(1.0) } }
}

impl PqCodebookTrainer for AnisotropicTrainer {
    fn name(&self) -> &'static str { "anisotropic-pq" }

    fn train(&self, train: &[Vec<f32>], cfg: &PqTrainConfig) -> Result<PqCodebook, PqError> {
        let n = train.len();
        if n < cfg.k {
            return Err(PqError::TooFewSamples { n, k: cfg.k });
        }
        let dim = train[0].len();
        let d = split_shape(dim, cfg.m)?;
        let mut centroids = vec![0.0f32; cfg.m * cfg.k * d];
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let h_orth = 1.0f32;
        let h_par = self.eta;

        for sub in 0..cfg.m {
            // Sub-vectors and per-sub-space unit directions. For MIPS the
            // ranking-critical residual is the one *aligned with the datum's
            // own direction inside the sub-space*: pushing a code centroid
            // along that axis directly changes the inner-product score
            // between the datum and any query, while orthogonal moves average
            // out. We therefore weight parallel residual energy by η.
            let sv: Vec<Vec<f32>> = train
                .iter()
                .map(|x| x[sub * d..(sub + 1) * d].to_vec())
                .collect();
            let dirs: Vec<Option<Vec<f32>>> = sv
                .iter()
                .map(|x| {
                    let n2 = norm_sq(x);
                    if n2 <= f32::EPSILON {
                        None
                    } else {
                        let inv = 1.0 / n2.sqrt();
                        Some(x.iter().map(|v| v * inv).collect())
                    }
                })
                .collect();
            let mut centers = kmeans_pp_init(&sv, cfg.k, &mut rng);

            for _ in 0..cfg.iters {
                // Assignment (weighted L2 to centers).
                let mut assign = vec![0usize; sv.len()];
                for (i, x) in sv.iter().enumerate() {
                    let u = &dirs[i];
                    let mut best = 0usize;
                    let mut best_v = f32::INFINITY;
                    for (c, cv) in centers.iter().enumerate() {
                        let v = weighted_residual_sq(x, cv, u.as_deref(), h_par, h_orth);
                        if v < best_v {
                            best_v = v;
                            best = c;
                        }
                    }
                    assign[i] = best;
                }
                // Update: solve (Σ Mᵢ) c = Σ Mᵢ xᵢ per cluster.
                for c in 0..cfg.k {
                    let members: Vec<usize> = assign
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &a)| if a == c { Some(i) } else { None })
                        .collect();
                    if members.is_empty() {
                        continue;
                    }
                    let new_c =
                        anisotropic_centroid(&sv, &dirs, &members, d, h_par, h_orth);
                    centers[c] = new_c;
                }
            }

            let base = sub * cfg.k * d;
            for c in 0..cfg.k {
                centroids[base + c * d..base + (c + 1) * d].copy_from_slice(&centers[c]);
            }
        }
        Ok(PqCodebook { m: cfg.m, k: cfg.k, d, centroids })
    }
}

fn weighted_residual_sq(
    x: &[f32],
    c: &[f32],
    u: Option<&[f32]>,
    h_par: f32,
    h_orth: f32,
) -> f32 {
    // r = x - c ;  ||r||² = r · r  ;  (r·u)² is the parallel-magnitude sq.
    let mut rr = 0.0f32;
    let mut ru = 0.0f32;
    if let Some(u) = u {
        for j in 0..x.len() {
            let r = x[j] - c[j];
            rr += r * r;
            ru += r * u[j];
        }
        let par = ru * ru;
        let orth = (rr - par).max(0.0);
        h_par * par + h_orth * orth
    } else {
        h_orth * l2_sq(x, c)
    }
}

/// Solve (Σ Mᵢ) c = Σ Mᵢ xᵢ where Mᵢ = h_orth I + (h_par - h_orth) uᵢuᵢᵀ.
/// Uses naive Gauss-Jordan on a `d × d` matrix (d ≤ 32 in practice).
fn anisotropic_centroid(
    sv: &[Vec<f32>],
    dirs: &[Option<Vec<f32>>],
    members: &[usize],
    d: usize,
    h_par: f32,
    h_orth: f32,
) -> Vec<f32> {
    let alpha = h_par - h_orth;
    let mut a = vec![0.0f64; d * d];
    let mut b = vec![0.0f64; d];
    let base_diag = h_orth as f64 * members.len() as f64;
    for j in 0..d {
        a[j * d + j] += base_diag;
    }
    for &i in members {
        let x = &sv[i];
        if let Some(u) = dirs[i].as_deref() {
            // Add α uuᵀ to A, plus h_orth I already accounted for above.
            for j in 0..d {
                let uj = u[j] as f64;
                for l in 0..d {
                    a[j * d + l] += (alpha as f64) * uj * (u[l] as f64);
                }
                // b += Mᵢ xᵢ  = h_orth x + α (u·x) u
                b[j] += (h_orth as f64) * (x[j] as f64);
            }
            let mut ux = 0.0f64;
            for j in 0..d {
                ux += u[j] as f64 * x[j] as f64;
            }
            for j in 0..d {
                b[j] += (alpha as f64) * ux * (u[j] as f64);
            }
        } else {
            for j in 0..d {
                b[j] += (h_orth as f64) * (x[j] as f64);
            }
        }
    }
    // Gauss-Jordan elimination (partial pivoting).
    for k in 0..d {
        // Pivot.
        let mut piv = k;
        let mut piv_v = a[k * d + k].abs();
        for r in (k + 1)..d {
            let v = a[r * d + k].abs();
            if v > piv_v {
                piv_v = v;
                piv = r;
            }
        }
        if piv_v <= 1e-12 {
            // Singular; fall back to isotropic mean for numerical safety.
            let mut fallback = vec![0.0f32; d];
            for &i in members {
                for j in 0..d {
                    fallback[j] += sv[i][j];
                }
            }
            let inv = 1.0 / members.len() as f32;
            for j in 0..d {
                fallback[j] *= inv;
            }
            return fallback;
        }
        if piv != k {
            for j in 0..d {
                a.swap(k * d + j, piv * d + j);
            }
            b.swap(k, piv);
        }
        let inv = 1.0 / a[k * d + k];
        for j in 0..d {
            a[k * d + j] *= inv;
        }
        b[k] *= inv;
        for r in 0..d {
            if r == k {
                continue;
            }
            let f = a[r * d + k];
            if f == 0.0 {
                continue;
            }
            for j in 0..d {
                a[r * d + j] -= f * a[k * d + j];
            }
            b[r] -= f * b[k];
        }
    }
    b.into_iter().map(|v| v as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn isotropic_trains_reasonable_codebook() {
        let data = synth(512, 16, 1);
        let cfg = PqTrainConfig { m: 4, k: 16, iters: 8, seed: 1 };
        let cb = IsotropicTrainer.train(&data, &cfg).unwrap();
        assert_eq!(cb.centroids.len(), cfg.m * cfg.k * cb.d);
        assert_eq!(cb.d, 4);
    }

    #[test]
    fn anisotropic_beats_isotropic_on_mips_residual() {
        // Anisotropic training should reduce parallel-residual energy.
        let data = synth(1024, 16, 7);
        let cfg = PqTrainConfig { m: 4, k: 32, iters: 10, seed: 7 };
        let iso = IsotropicTrainer.train(&data, &cfg).unwrap();
        let ani = AnisotropicTrainer::new(8.0).train(&data, &cfg).unwrap();

        let par_energy = |cb: &PqCodebook| -> f32 {
            let mut total = 0.0f32;
            let mut buf = vec![0u8; cb.m];
            for x in &data {
                cb.encode(x, &mut buf);
                for sub in 0..cb.m {
                    let sv = &x[sub * cb.d..(sub + 1) * cb.d];
                    let n2 = norm_sq(sv);
                    if n2 <= f32::EPSILON { continue; }
                    let u: Vec<f32> = sv.iter().map(|v| v / n2.sqrt()).collect();
                    let c = cb.centroid(sub, buf[sub] as usize);
                    let mut ru = 0.0;
                    for j in 0..cb.d { ru += (sv[j] - c[j]) * u[j]; }
                    total += ru * ru;
                }
            }
            total
        };
        let iso_par = par_energy(&iso);
        let ani_par = par_energy(&ani);
        assert!(
            ani_par < iso_par,
            "anisotropic parallel energy {ani_par} should beat isotropic {iso_par}"
        );
    }
}
