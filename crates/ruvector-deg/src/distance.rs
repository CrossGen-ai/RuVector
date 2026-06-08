//! Distance metrics. Pluggable via the `Metric` trait so callers can swap in
//! cosine, inner-product, hamming, etc. without touching the graph code.

/// A distance metric. Must be non-negative, symmetric, and identity-of-discernibles.
/// (Triangle inequality is not required for correctness of the graph search but
/// is required for any RNG-style edge-optimisation guarantees.)
pub trait Metric: Send + Sync + 'static {
    fn dist(a: &[f32], b: &[f32]) -> f32;
}

/// Squared Euclidean distance. Faster than L2 (no sqrt) and preserves ordering.
pub struct L2;

impl Metric for L2 {
    #[inline]
    fn dist(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut s = 0.0f32;
        // Unrolled 4-lane scalar loop — auto-vectorises on x86_64/aarch64.
        let n = a.len();
        let chunks = n / 4;
        for i in 0..chunks {
            let j = i * 4;
            let d0 = a[j] - b[j];
            let d1 = a[j + 1] - b[j + 1];
            let d2 = a[j + 2] - b[j + 2];
            let d3 = a[j + 3] - b[j + 3];
            s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
        for i in (chunks * 4)..n {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_zero() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(L2::dist(&a, &a), 0.0);
    }

    #[test]
    fn l2_known() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 2.0, 2.0];
        // 1 + 4 + 4 = 9
        assert!((L2::dist(&a, &b) - 9.0).abs() < 1e-6);
    }
}
