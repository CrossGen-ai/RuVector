//! Minimal vector math on `&[f32]`. No `unsafe`, no SIMD intrinsics — the
//! compiler auto-vectorizes these tight loops in `--release` on x86_64
//! (verified via `cargo asm`). Kept trivial so the benchmark numbers reflect
//! algorithmic choices, not micro-tuned kernels.

/// Squared Euclidean distance.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_sq: dim mismatch");
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Standard dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: dim mismatch");
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// `out[i] = a[i] - b[i]`.
#[inline]
pub fn sub_into(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "sub_into: a/b dim mismatch");
    assert_eq!(a.len(), out.len(), "sub_into: out dim mismatch");
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

/// L2 norm.
#[inline]
pub fn norm(a: &[f32]) -> f32 {
    l2_sq(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_basic() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 2.0];
        assert!((l2_sq(&a, &b) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn dot_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        // 4 + 10 + 18 = 32
        assert!((dot(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn sub_into_basic() {
        let mut out = [0.0f32; 3];
        let a = [5.0f32, 5.0, 5.0];
        let b = [1.0f32, 2.0, 3.0];
        sub_into(&mut out, &a, &b);
        assert_eq!(out, [4.0, 3.0, 2.0]);
    }

    #[test]
    #[should_panic]
    fn l2_sq_mismatch_panics() {
        let a = [0.0f32; 3];
        let b = [0.0f32; 4];
        l2_sq(&a, &b);
    }
}
