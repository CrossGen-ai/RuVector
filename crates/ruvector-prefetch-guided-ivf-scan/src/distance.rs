//! L2² distance kernels.
//!
//! We stay in stable Rust and let the autovectorizer handle SIMD. On aarch64
//! this lowers to NEON `FMLA`/`FSUB` pipes; on x86_64 to AVX/AVX2 when
//! `target-cpu=native` is used. The point of the prefetch benchmark isn't to
//! win on the ALU side, it's to hide DRAM latency for the *next* vector.
//! Keeping the kernel simple makes the memory-bound regime easy to see.

/// Squared L2 distance. Panics in debug if lengths differ; returns junk in
/// release if they differ (the caller is expected to have validated).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2_sq length mismatch");
    let n = a.len().min(b.len());
    let mut acc = 0.0f32;
    // Manual 4-way unroll — the autovectorizer picks it up as NEON on M-series
    // and AVX on x86_64. Chunked loop plus scalar tail.
    let mut i = 0;
    let chunks = n / 4;
    while i < chunks * 4 {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < n {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_distance_when_equal() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(l2_sq(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn simple_distance() {
        let a = [0.0f32, 0.0, 0.0];
        let b = [1.0f32, 2.0, 2.0];
        // 1 + 4 + 4 = 9
        assert!((l2_sq(&a, &b) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn odd_length_tail() {
        let a = vec![1.0f32; 7];
        let b = vec![0.0f32; 7];
        assert!((l2_sq(&a, &b) - 7.0).abs() < 1e-5);
    }
}
