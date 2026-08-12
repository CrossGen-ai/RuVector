//! SOAR-IVF: Spilling with Orthogonality-Amplified Residuals for IVF.
//!
//! Three measurable variants under a shared `IvfVariant` trait:
//!   - `IvfSingle`    – classic IVF-Flat, one centroid per vector.
//!   - `IvfSpillTopK` – naive spilling: assign each vector to its top-`s`
//!                      nearest centroids (baseline for SOAR to beat).
//!   - `IvfSoar`      – SOAR: primary = nearest centroid; secondary chosen to
//!                      minimise the *orthogonality-amplified* residual cost
//!                      `‖r_s‖² + λ · ((r_p · r_s)² / ‖r_p‖²)` where
//!                      `r_p = x − c_p` and `r_s = x − c_s`. Amplifies the
//!                      component of the secondary residual that is *parallel*
//!                      to the primary residual, biasing the spilled copy
//!                      toward centroids that cover error directions the
//!                      primary poorly represents.
//!
//! All three share the same k-means-lite centroid trainer and the same
//! IVF-Flat probe-and-scan retrieval; only the assignment step differs, so
//! recall differences are attributable to SOAR alone.
//!
//! No external deps. `f32` corpus, squared-L2 distance.

pub mod dataset;
pub mod ivf_common;
pub mod ivf_single;
pub mod ivf_spill_topk;
pub mod ivf_soar;
pub mod metrics;

pub use ivf_common::{Centroids, IvfVariant, PostingLists};
pub use ivf_single::IvfSingle;
pub use ivf_soar::IvfSoar;
pub use ivf_spill_topk::IvfSpillTopK;
pub use metrics::{recall_at_k, Hit};

/// Squared L2 distance for f32 slices.
#[inline(always)]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Dot product for f32 slices.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// In-place: `out[i] = a[i] - b[i]`.
#[inline]
pub fn sub_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_l2_self_zero() {
        let v = vec![1.0f32, -2.0, 3.5];
        assert!(sq_l2(&v, &v) < 1e-9);
    }

    #[test]
    fn sq_l2_symmetric() {
        let a = vec![0.0f32, 1.0, 2.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((sq_l2(&a, &b) - sq_l2(&b, &a)).abs() < 1e-9);
    }

    #[test]
    fn dot_basic() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, -1.0, 2.0];
        // 4 - 2 + 6 = 8
        assert!((dot(&a, &b) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn sub_into_basic() {
        let a = vec![5.0f32, 3.0, 1.0];
        let b = vec![1.0f32, 1.0, 1.0];
        let mut out = vec![0.0f32; 3];
        sub_into(&a, &b, &mut out);
        assert_eq!(out, vec![4.0, 2.0, 0.0]);
    }
}
