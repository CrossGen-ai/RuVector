//! Distance primitives and search-time counters.

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

/// Per-search counters. Instrumenting the walk is how we measure "did the
/// seed save work", decoupled from wall-clock noise.
#[derive(Default, Debug, Clone, Copy)]
pub struct SearchStats {
    pub hops: u64,
    pub distance_computations: u64,
    pub entry_distance: f32,
    pub recall_hits: u32,
}
