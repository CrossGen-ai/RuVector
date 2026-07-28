//! Deterministic orthonormal rotation used to make the partial-sum of a
//! squared L2 distance an unbiased estimator of the full distance.
//!
//! We construct the rotation as a product of two Householder reflections
//! seeded from a supplied `u64`. Two reflections suffice to give a
//! matrix that is (a) orthonormal by construction, (b) full-rank almost
//! surely, and (c) mixes every input coordinate into every output
//! coordinate — which is the only property ADSampling actually depends
//! on (Lemma 3.1 of the paper).
//!
//! No `nalgebra` / BLAS dependency: rotations run in O(d) per apply.

use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Deterministic orthonormal rotation R applied as v ↦ Rv.
#[derive(Clone, Debug)]
pub struct RandomRotation {
    d: usize,
    // Two Householder reflection vectors, each of unit L2 norm.
    h1: Vec<f32>,
    h2: Vec<f32>,
}

impl RandomRotation {
    /// Build a rotation for `d`-dimensional vectors from a `seed`.
    pub fn new(d: usize, seed: u64) -> Self {
        assert!(d >= 2, "rotation dimension must be >= 2");
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let h1 = sample_unit(&mut rng, d);
        let h2 = sample_unit(&mut rng, d);
        Self { d, h1, h2 }
    }

    /// Dimensionality this rotation was built for.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Return R * v as a fresh vector.
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.d, "vector dim mismatch");
        let mut out = v.to_vec();
        householder_in_place(&mut out, &self.h1);
        householder_in_place(&mut out, &self.h2);
        out
    }
}

fn sample_unit(rng: &mut impl Rng, d: usize) -> Vec<f32> {
    // A random Householder vector: sample from a d-dim Gaussian then
    // normalize.  Rejects the (measure-zero) all-zero draw.
    loop {
        let v: Vec<f32> = (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect();
        let n2: f32 = v.iter().map(|x| x * x).sum();
        if n2 > 1e-12 {
            let inv = 1.0 / n2.sqrt();
            return v.into_iter().map(|x| x * inv).collect();
        }
    }
}

// Applies (I - 2 h h^T) v in place. h must be unit-norm.
fn householder_in_place(v: &mut [f32], h: &[f32]) {
    debug_assert_eq!(v.len(), h.len());
    let mut dot = 0.0f32;
    for i in 0..v.len() {
        dot += v[i] * h[i];
    }
    let scale = 2.0 * dot;
    for i in 0..v.len() {
        v[i] -= scale * h[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_preserves_norm() {
        let r = RandomRotation::new(64, 0xC0FFEE);
        let v: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let rv = r.apply(&v);
        let n_v: f32 = v.iter().map(|x| x * x).sum();
        let n_rv: f32 = rv.iter().map(|x| x * x).sum();
        assert!((n_v - n_rv).abs() < 1e-3, "norm changed: {n_v} vs {n_rv}");
    }

    #[test]
    fn rotation_is_deterministic() {
        let r1 = RandomRotation::new(32, 42);
        let r2 = RandomRotation::new(32, 42);
        let v: Vec<f32> = (0..32).map(|i| i as f32).collect();
        assert_eq!(r1.apply(&v), r2.apply(&v));
    }

    #[test]
    fn rotation_preserves_pairwise_distance() {
        let r = RandomRotation::new(48, 7);
        let a: Vec<f32> = (0..48).map(|i| (i as f32) * 0.5).collect();
        let b: Vec<f32> = (0..48).map(|i| (i as f32 + 1.0) * 0.25).collect();
        let ra = r.apply(&a);
        let rb = r.apply(&b);
        let d_orig: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
        let d_rot: f32 = ra.iter().zip(&rb).map(|(x, y)| (x - y).powi(2)).sum();
        assert!((d_orig - d_rot).abs() < 1e-2, "{d_orig} vs {d_rot}");
    }
}
