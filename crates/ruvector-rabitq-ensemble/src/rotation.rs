//! Three rotation backends. All are deterministic given `(dim, seed, kind)`.
//!
//! * `HaarDense`      — true Haar-uniform `D×D` matrix via Gram–Schmidt on
//!                      a Gaussian block. `O(D²)` per apply; stores `4·D²` B.
//! * `HadamardSigned` — `D₁·H·D₂·H·D₃` with three independent ±1 diagonals
//!                      and the Fast Walsh–Hadamard Transform in between.
//!                      `O(D log D)` per apply; stores `3·D_pad` floats.
//!                      This matches TurboQuant's RHT (arXiv:2504.19874 §3.2)
//!                      and is "close enough to Haar" for RaBitQ's JL bound.
//! * `HadamardSingle` — `D·H` with one ±1 diagonal and one FWHT.
//!                      `O(D log D)`; stores `D_pad` floats. Cheaper, slightly
//!                      worse pseudo-orthogonality.
//!
//! All three normalise by `1/√D_pad` so that `(R x)·(R x) = x·x` (up to
//! padding) and the RaBitQ unbiasedness proof carries through.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationKind {
    HaarDense,
    HadamardSigned,
    HadamardSingle,
}

#[derive(Clone)]
enum Mode {
    HaarDense {
        matrix: Vec<f32>, // row-major D×D
    },
    HadamardSigned {
        signs: [Vec<f32>; 3], // each padded_dim
        padded_dim: usize,
    },
    HadamardSingle {
        signs: Vec<f32>, // padded_dim
        padded_dim: usize,
    },
}

#[derive(Clone)]
pub struct Rotation {
    mode: Mode,
    pub dim: usize,
    pub kind: RotationKind,
}

impl Rotation {
    pub fn new(dim: usize, kind: RotationKind, seed: u64) -> Self {
        assert!(dim > 0, "dim must be > 0");
        let mut rng = StdRng::seed_from_u64(seed);
        let mode = match kind {
            RotationKind::HaarDense => Mode::HaarDense {
                matrix: gram_schmidt_haar(dim, &mut rng),
            },
            RotationKind::HadamardSigned => {
                let padded_dim = dim.next_power_of_two().max(2);
                let signs: [Vec<f32>; 3] = [
                    sample_signs(padded_dim, &mut rng),
                    sample_signs(padded_dim, &mut rng),
                    sample_signs(padded_dim, &mut rng),
                ];
                Mode::HadamardSigned { signs, padded_dim }
            }
            RotationKind::HadamardSingle => {
                let padded_dim = dim.next_power_of_two().max(2);
                let signs = sample_signs(padded_dim, &mut rng);
                Mode::HadamardSingle { signs, padded_dim }
            }
        };
        Self { mode, dim, kind }
    }

    /// Output length depends on the backend: dense is `dim`, Hadamard
    /// variants are `padded_dim ≥ dim`. We expose this so callers can size
    /// their bit buffers correctly.
    pub fn out_dim(&self) -> usize {
        match &self.mode {
            Mode::HaarDense { .. } => self.dim,
            Mode::HadamardSigned { padded_dim, .. } => *padded_dim,
            Mode::HadamardSingle { padded_dim, .. } => *padded_dim,
        }
    }

    pub fn aux_bytes(&self) -> usize {
        match &self.mode {
            Mode::HaarDense { matrix } => matrix.len() * 4,
            Mode::HadamardSigned { signs, .. } => signs.iter().map(|s| s.len() * 4).sum(),
            Mode::HadamardSingle { signs, .. } => signs.len() * 4,
        }
    }

    /// Apply R to `x` (len == dim). Writes `out_dim()` floats into `out`.
    pub fn apply_into(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.dim, "input dim mismatch");
        assert_eq!(out.len(), self.out_dim(), "output dim mismatch");
        match &self.mode {
            Mode::HaarDense { matrix } => {
                let d = self.dim;
                for i in 0..d {
                    let row = &matrix[i * d..(i + 1) * d];
                    let mut acc = 0.0_f32;
                    for j in 0..d {
                        acc += row[j] * x[j];
                    }
                    out[i] = acc;
                }
            }
            Mode::HadamardSigned { signs, padded_dim } => {
                // zero-pad into out
                for v in out.iter_mut() {
                    *v = 0.0;
                }
                out[..self.dim].copy_from_slice(x);
                // D3 · H · D2 · H · D1  (applied right-to-left on x)
                for i in 0..*padded_dim {
                    out[i] *= signs[0][i];
                }
                fwht_in_place(out);
                for i in 0..*padded_dim {
                    out[i] *= signs[1][i];
                }
                fwht_in_place(out);
                for i in 0..*padded_dim {
                    out[i] *= signs[2][i];
                }
                // Normalise so that two FWHTs don't inflate scale by 2^log2(n).
                // One in-place FWHT with "divide by sqrt(n) at end" per transform
                // means two transforms scale by n total — divide by n here.
                let norm = (*padded_dim as f32).recip();
                for v in out.iter_mut() {
                    *v *= norm;
                }
            }
            Mode::HadamardSingle { signs, padded_dim } => {
                for v in out.iter_mut() {
                    *v = 0.0;
                }
                out[..self.dim].copy_from_slice(x);
                for i in 0..*padded_dim {
                    out[i] *= signs[i];
                }
                fwht_in_place(out);
                // One FWHT: divide by sqrt(n) to preserve ‖·‖₂.
                let norm = (*padded_dim as f32).sqrt().recip();
                for v in out.iter_mut() {
                    *v *= norm;
                }
            }
        }
    }

    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.out_dim()];
        self.apply_into(x, &mut out);
        out
    }
}

fn sample_signs(n: usize, rng: &mut StdRng) -> Vec<f32> {
    (0..n).map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect()
}

/// In-place iterative Fast Walsh–Hadamard Transform (unscaled).
/// Length MUST be a power of two.
pub(crate) fn fwht_in_place(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

/// Haar-uniform `D×D` orthogonal matrix via Gram–Schmidt on a Gaussian
/// block. Deterministic given the rng. Return value is row-major `D·D`.
fn gram_schmidt_haar(d: usize, rng: &mut StdRng) -> Vec<f32> {
    let mut m = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let g: f32 = StandardNormal.sample(rng);
            m[i * d + j] = g;
        }
    }
    for i in 0..d {
        // Subtract projections onto previously orthonormalised rows.
        for p in 0..i {
            let (first, rest) = m.split_at_mut(i * d);
            let row_p = &first[p * d..(p + 1) * d];
            let row_i = &mut rest[..d];
            let mut dot = 0.0_f32;
            for k in 0..d {
                dot += row_p[k] * row_i[k];
            }
            for k in 0..d {
                row_i[k] -= dot * row_p[k];
            }
        }
        // Normalise.
        let row_i = &mut m[i * d..(i + 1) * d];
        let mut nrm = 0.0_f32;
        for k in 0..d {
            nrm += row_i[k] * row_i[k];
        }
        let inv = nrm.sqrt().recip();
        for k in 0..d {
            row_i[k] *= inv;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_roundtrip_power_of_two() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0_f32];
        let orig = a.clone();
        fwht_in_place(&mut a);
        fwht_in_place(&mut a);
        for (x, y) in a.iter().zip(orig.iter()) {
            assert!((*x / 8.0 - *y).abs() < 1e-5, "fwht roundtrip: {x} vs {y}");
        }
    }

    #[test]
    fn haar_rotation_preserves_norm() {
        let r = Rotation::new(16, RotationKind::HaarDense, 7);
        let x: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let y = r.apply(&x);
        let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((nx - ny).abs() < 1e-4, "{nx} vs {ny}");
    }

    #[test]
    fn hadamard_signed_preserves_norm_approx() {
        let r = Rotation::new(17, RotationKind::HadamardSigned, 42);
        let x: Vec<f32> = (0..17).map(|i| (i as f32 * 0.3).cos()).collect();
        let y = r.apply(&x);
        let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        // padded to 32; two FWHTs normalised by 1/n keep the energy near the
        // zero-padded input's energy, which equals the original.
        assert!(
            (nx - ny).abs() < 5e-3,
            "signed Hadamard norm drift: {nx} vs {ny}"
        );
    }

    #[test]
    fn hadamard_single_preserves_norm() {
        let r = Rotation::new(16, RotationKind::HadamardSingle, 11);
        let x: Vec<f32> = (0..16).map(|i| (i as f32 * 0.7).sin()).collect();
        let y = r.apply(&x);
        let nx: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let ny: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((nx - ny).abs() < 1e-4, "single Hadamard: {nx} vs {ny}");
    }

    #[test]
    fn determinism() {
        let r1 = Rotation::new(32, RotationKind::HadamardSigned, 123);
        let r2 = Rotation::new(32, RotationKind::HadamardSigned, 123);
        let x: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(r1.apply(&x), r2.apply(&x));
    }
}
