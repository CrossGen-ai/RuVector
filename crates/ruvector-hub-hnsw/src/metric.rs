//! Distance metrics. Currently squared L2 (monotone with L2, so ordering preserved).

pub type Vector = Vec<f32>;

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn l2_sq_basic() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 2.0];
        assert!((l2_sq(&a, &b) - 9.0).abs() < 1e-6);
    }
    #[test]
    fn l2_sq_zero() {
        let a = vec![0.5, -0.5, 1.0];
        assert!(l2_sq(&a, &a) < 1e-6);
    }
}
