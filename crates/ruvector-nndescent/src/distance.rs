//! L2-squared distance with a thread-safe operation counter.

use std::sync::atomic::{AtomicU64, Ordering};

#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut s = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Tracks how many distance computations a builder performed.
/// Used as an algorithm-fairness metric across variants (independent
/// of SIMD intrinsics or cache behavior).
#[derive(Default, Debug)]
pub struct DistanceCounter {
    n: AtomicU64,
}

impl DistanceCounter {
    pub fn new() -> Self {
        Self { n: AtomicU64::new(0) }
    }
    #[inline(always)]
    pub fn measure(&self, a: &[f32], b: &[f32]) -> f32 {
        self.n.fetch_add(1, Ordering::Relaxed);
        l2_sq(a, b)
    }
    pub fn count(&self) -> u64 {
        self.n.load(Ordering::Relaxed)
    }
    pub fn reset(&self) {
        self.n.store(0, Ordering::Relaxed);
    }
}
