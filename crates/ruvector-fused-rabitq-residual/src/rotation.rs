//! Signed Fast Walsh–Hadamard Transform (SFHT): a seeded, L2-preserving
//! O(D log D) orthogonal rotation without any matrix storage.
//!
//! Given random diagonal signs `S ∈ {±1}^D` and the normalized Hadamard
//! matrix `H_norm = H / sqrt(D)` (so `H_norm H_normᵀ = I`), the transform
//! `x ↦ H_norm · (S · x)` is orthogonal. It's the standard building block
//! in RaBitQ (Gao & Long 2024) and in the Fast Johnson–Lindenstrauss
//! Transform (Ailon & Chazelle 2009): it spreads energy uniformly across
//! dimensions so 1-bit sign codes become quasi-isotropic.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct SignedHadamard {
    pub dim: usize,
    signs: Vec<f32>,
    scale: f32,
}

impl SignedHadamard {
    /// Build a seeded SFHT for a power-of-two dimension.
    pub fn new_seeded(dim: usize, seed: u64) -> Self {
        assert!(dim.is_power_of_two(), "SFHT dim must be power of two");
        let mut rng = StdRng::seed_from_u64(seed);
        let signs: Vec<f32> = (0..dim)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let scale = 1.0 / (dim as f32).sqrt();
        Self { dim, signs, scale }
    }

    /// Apply the rotation in place. L2-preserving.
    pub fn apply(&self, v: &mut [f32]) {
        assert_eq!(v.len(), self.dim);
        for i in 0..self.dim {
            v[i] *= self.signs[i];
        }
        fwht(v);
        for x in v.iter_mut() {
            *x *= self.scale;
        }
    }
}

/// In-place iterative Fast Walsh–Hadamard Transform (Sylvester form).
/// Result satisfies `H · Hᵀ = D · I`; caller normalizes by `1/sqrt(D)`.
fn fwht(a: &mut [f32]) {
    let n = a.len();
    let mut h = 1;
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
