//! Fast approximate random rotation via sign-flipped Walsh-Hadamard.
//!
//! P = D_s * H where H is Hadamard (sign-only, O(D log D)) and D_s is
//! a diagonal of random ±1. Preserves norms exactly and distributes
//! energy uniformly across coordinates, matching the Gaussian
//! marginal assumption used by the eRaBitQ codebook.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

pub trait Rotation {
    fn apply(&self, x: &mut [f32]);
    fn dim(&self) -> usize;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FastRotation {
    dim: usize,
    padded: usize,
    signs: Vec<i8>,
}

impl FastRotation {
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded = dim.next_power_of_two().max(2);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let signs = (0..padded)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect();
        Self { dim, padded, signs }
    }

    pub fn padded_dim(&self) -> usize {
        self.padded
    }

    /// Apply in-place with a zero-padded working buffer.
    pub fn apply_padded(&self, buf: &mut [f32]) {
        assert_eq!(buf.len(), self.padded);
        for (b, s) in buf.iter_mut().zip(&self.signs) {
            *b *= *s as f32;
        }
        fwht(buf);
        // normalise so that H is orthogonal (Hadamard scales by sqrt(n))
        let norm = (self.padded as f32).sqrt();
        for b in buf.iter_mut() {
            *b /= norm;
        }
    }
}

impl Rotation for FastRotation {
    fn apply(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.dim);
        let mut buf = vec![0.0f32; self.padded];
        buf[..self.dim].copy_from_slice(x);
        self.apply_padded(&mut buf);
        x.copy_from_slice(&buf[..self.dim]);
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

/// In-place Fast Walsh-Hadamard transform, length must be a power of 2.
pub fn fwht(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
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

/// Rotate into a freshly allocated buffer of size `padded_dim`.
pub fn rotate_to_padded(rot: &FastRotation, x: &[f32]) -> Vec<f32> {
    let mut buf = vec![0.0f32; rot.padded_dim()];
    buf[..x.len()].copy_from_slice(x);
    rot.apply_padded(&mut buf);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_preserves_norm() {
        let mut v = vec![1.0f32, 2.0, -1.0, 3.0, 0.5, -2.5, 1.5, 0.0];
        let n2_before: f32 = v.iter().map(|x| x * x).sum();
        fwht(&mut v);
        // Hadamard without normalisation multiplies norm by sqrt(n).
        let n2_after: f32 = v.iter().map(|x| x * x).sum::<f32>() / (v.len() as f32);
        assert!((n2_before - n2_after).abs() < 1e-4);
    }

    #[test]
    fn rotation_preserves_norm() {
        let rot = FastRotation::new(64, 42);
        let x: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let n2: f32 = x.iter().map(|v| v * v).sum();
        let r = rotate_to_padded(&rot, &x);
        let n2r: f32 = r.iter().map(|v| v * v).sum();
        assert!((n2 - n2r).abs() < 1e-3, "{} vs {}", n2, n2r);
    }
}
