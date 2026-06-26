//! Distance metrics. Squared L2 only — sufficient for ranking and avoids
//! the sqrt in every hop of greedy search.

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    // Hand-unrolled by 4 for autovectorization on stable rustc.
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let i = i * 4;
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for i in (chunks * 4)..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sq_l2_basic() {
        assert_eq!(sq_l2(&[0.0; 4], &[1.0; 4]), 4.0);
        assert!((sq_l2(&[1.0, 2.0, 3.0], &[4.0, 6.0, 8.0]) - (9.0 + 16.0 + 25.0)).abs() < 1e-5);
    }
}
