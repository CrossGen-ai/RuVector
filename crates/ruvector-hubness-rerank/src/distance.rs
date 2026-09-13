//! Distance trait and default L2 implementation.

/// Any real-valued dissimilarity. Must be non-negative and symmetric for the
/// rerank stats to be meaningful; a full metric is not required.
pub trait Distance: Send + Sync {
    fn dist(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Squared-Euclidean would rank identically to Euclidean, but hubness stats
/// mix distances additively/multiplicatively, so we keep the true L2 metric
/// here for interpretability.
#[derive(Clone, Copy, Debug, Default)]
pub struct L2;

impl Distance for L2 {
    #[inline]
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // Manual unroll-friendly loop; the release profile vectorises this
        // to SSE/AVX on x86_64 and NEON on aarch64.
        let mut s = 0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_basic() {
        let a = [0f32, 0.0, 0.0];
        let b = [3f32, 4.0, 0.0];
        assert!((L2.dist(&a, &b) - 5.0).abs() < 1e-6);
        assert_eq!(L2.dist(&a, &a), 0.0);
        assert!((L2.dist(&a, &b) - L2.dist(&b, &a)).abs() < 1e-6);
    }
}
