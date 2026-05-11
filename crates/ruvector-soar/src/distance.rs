//! Distance primitives. Plain Rust, autovectorized; no `unsafe`, no SIMD intrinsics.

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[inline]
pub fn norm_sq(a: &[f32]) -> f32 {
    dot(a, a)
}

/// Compute residual `x - c` into `out`.
#[inline]
pub fn residual(x: &[f32], c: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), c.len());
    debug_assert_eq!(out.len(), c.len());
    for i in 0..x.len() {
        out[i] = x[i] - c[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_zero() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(l2_sq(&a, &a), 0.0);
    }

    #[test]
    fn l2_basic() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((l2_sq(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn dot_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-6);
    }
}
