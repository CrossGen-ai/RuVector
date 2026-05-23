//! Distance primitives for `ruvector-roargraph`.
//!
//! Kept intentionally tiny — vectors are `Vec<f32>`, distance is squared L2.
//! Squared L2 is monotone in L2 so it's a valid kNN metric, and it avoids the
//! sqrt cost inside hot loops.

pub type Vector = Vec<f32>;

/// Squared L2 distance. Auto-vectorised by `rustc` on `f32` slices when both
/// inputs are aligned `Vec<f32>`; the explicit chunked accumulator nudges LLVM
/// toward SIMD even on stable.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dim mismatch in l2_sq");
    let n = a.len();
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        let d0 = a[j] - b[j];
        let d1 = a[j + 1] - b[j + 1];
        let d2 = a[j + 2] - b[j + 2];
        let d3 = a[j + 3] - b[j + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symmetric_and_zero_self() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 0.0, 1.0, 4.0, 5.0];
        assert!((l2_sq(&a, &a)).abs() < 1e-6);
        assert!((l2_sq(&a, &b) - l2_sq(&b, &a)).abs() < 1e-6);
        // (1)^2 + (2)^2 + (2)^2 + 0 + 0 = 9
        assert!((l2_sq(&a, &b) - 9.0).abs() < 1e-6);
    }
}
