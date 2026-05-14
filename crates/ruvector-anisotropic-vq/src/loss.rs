//! Anisotropic loss decomposition.
//!
//! For a unit-norm reference x and residual r = x - q, decompose:
//!   r_parallel      = (r . x) x         (component along x)
//!   r_perpendicular = r - r_parallel    (orthogonal component)
//!
//! Anisotropic loss with weight eta >= 1:
//!   L(x, q; eta) = eta * ||r_parallel||^2 + ||r_perpendicular||^2

#[derive(Clone, Copy, Debug)]
pub struct AnisotropicLoss {
    /// eta = 1 is plain MSE; eta > 1 weights parallel error more.
    pub eta: f32,
}

impl AnisotropicLoss {
    pub fn new(eta: f32) -> Self {
        assert!(eta >= 1.0, "eta must be >= 1.0 (eta=1 = isotropic/MSE)");
        Self { eta }
    }

    /// Score a single (x, q) pair. Lower is better.
    pub fn score(&self, x: &[f32], q: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), q.len());
        let (par, per) = decompose_residual(x, q);
        self.eta * par + per
    }
}

/// Returns `(||r_parallel||^2, ||r_perpendicular||^2)` where r = x - q
/// and the parallel direction is x (assumed approximately unit-norm).
///
/// For non-unit x, we project onto x_hat = x / ||x|| analytically. The
/// formulas are:
///   <r, x> = <x - q, x> = ||x||^2 - <q, x>
///   r_parallel_sq = (<r, x_hat>)^2 = (<r, x>)^2 / ||x||^2
///   r_perp_sq = ||r||^2 - r_parallel_sq
pub fn decompose_residual(x: &[f32], q: &[f32]) -> (f32, f32) {
    let mut rdotx = 0.0f32; // <r, x>
    let mut rnorm2 = 0.0f32; // ||r||^2
    let mut xnorm2 = 0.0f32; // ||x||^2
    for i in 0..x.len() {
        let xi = x[i];
        let ri = xi - q[i];
        rdotx += ri * xi;
        rnorm2 += ri * ri;
        xnorm2 += xi * xi;
    }
    if xnorm2 <= f32::EPSILON {
        return (0.0, rnorm2);
    }
    let par = (rdotx * rdotx) / xnorm2;
    let perp = (rnorm2 - par).max(0.0);
    (par, perp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposition_sums_to_residual_norm() {
        let x = [1.0f32, 0.0, 0.0];
        let q = [0.5, 0.5, 0.0];
        let (par, perp) = decompose_residual(&x, &q);
        // r = [0.5, -0.5, 0]; ||r||^2 = 0.5
        assert!((par + perp - 0.5).abs() < 1e-6);
        // Along x-axis only the x component contributes:
        assert!((par - 0.25).abs() < 1e-6);
        assert!((perp - 0.25).abs() < 1e-6);
    }

    #[test]
    fn eta_one_equals_mse() {
        let x = [0.6f32, 0.8, 0.0];
        let q = [0.55, 0.78, 0.05];
        let loss = AnisotropicLoss::new(1.0).score(&x, &q);
        let mse: f32 = x.iter().zip(&q).map(|(a, b)| (a - b).powi(2)).sum();
        assert!((loss - mse).abs() < 1e-6, "eta=1 must equal MSE");
    }

    #[test]
    fn eta_gt_one_amplifies_parallel() {
        let x = [1.0f32, 0.0];
        let q_parallel = [0.9, 0.0]; // pure parallel error
        let q_perp = [1.0, 0.1]; // pure perpendicular error
        let iso = AnisotropicLoss::new(1.0);
        let aniso = AnisotropicLoss::new(4.0);
        // Under MSE both errors are equal:
        assert!((iso.score(&x, &q_parallel) - iso.score(&x, &q_perp)).abs() < 1e-6);
        // Anisotropic must penalize parallel more:
        assert!(aniso.score(&x, &q_parallel) > aniso.score(&x, &q_perp));
    }
}
