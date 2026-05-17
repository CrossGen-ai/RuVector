//! Distance kernels. Plain Rust, autovectorizable; no `unsafe`.

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 { l2_sq(a, b).sqrt() }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((l2(&a, &b) - 5.0).abs() < 1e-6);
        assert!((l2_sq(&a, &b) - 25.0).abs() < 1e-6);
    }
}
