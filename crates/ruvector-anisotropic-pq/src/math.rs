//! Tiny linear algebra helpers used by the anisotropic k-means update.
//!
//! We solve small (typically d_sub ≤ 32) symmetric positive-definite systems
//! `A · x = b` per cluster per iteration. Cholesky decomposition suffices and
//! keeps the crate dependency-free.

/// Solve `A · x = b` in place for a symmetric positive-definite `d × d` matrix.
///
/// `a` is row-major and is destroyed. Returns `Some(x)` on success or `None`
/// if `A` was not sufficiently positive-definite (caller should fall back).
pub fn solve_spd(a: &mut [f64], b: &[f64], d: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), d * d);
    debug_assert_eq!(b.len(), d);

    // In-place Cholesky: A = L·Lᵀ, storing L in the lower triangle of `a`.
    for i in 0..d {
        for j in 0..=i {
            let mut sum = a[i * d + j];
            for k in 0..j {
                sum -= a[i * d + k] * a[j * d + k];
            }
            if i == j {
                if sum <= 1e-12 {
                    return None;
                }
                a[i * d + i] = sum.sqrt();
            } else {
                a[i * d + j] = sum / a[j * d + j];
            }
        }
    }

    // Forward: L · y = b
    let mut y = vec![0.0f64; d];
    for i in 0..d {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * d + k] * y[k];
        }
        y[i] = sum / a[i * d + i];
    }

    // Backward: Lᵀ · x = y
    let mut x = vec![0.0f64; d];
    for i in (0..d).rev() {
        let mut sum = y[i];
        for k in (i + 1)..d {
            sum -= a[k * d + i] * x[k];
        }
        x[i] = sum / a[i * d + i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_solve() {
        let mut a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 5.0];
        let x = solve_spd(&mut a, &b, 2).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-9);
        assert!((x[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn spd_solve_matches_hand_calc() {
        // A = [[4,2],[2,3]] (SPD), b = [10, 8] → x = [1.75, 1.5]
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let b = vec![10.0, 8.0];
        let x = solve_spd(&mut a, &b, 2).unwrap();
        assert!((x[0] - 1.75).abs() < 1e-6, "{}", x[0]);
        assert!((x[1] - 1.5).abs() < 1e-6, "{}", x[1]);
    }

    #[test]
    fn singular_returns_none() {
        // Zero matrix is not SPD.
        let mut a = vec![0.0; 4];
        let b = vec![1.0, 1.0];
        assert!(solve_spd(&mut a, &b, 2).is_none());
    }
}
