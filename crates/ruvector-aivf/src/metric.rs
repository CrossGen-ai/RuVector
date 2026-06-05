//! Distance primitives.  Kept tiny so SIMD lanes/portable_simd can replace
//! them later without touching the index logic.

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    let mut i = 0;
    // Manual 4-way unroll — most autovectorisers fuse this on x86_64.
    while i + 4 <= a.len() {
        let d0 = a[i]     - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        s += d0*d0 + d1*d1 + d2*d2 + d3*d3;
        i += 4;
    }
    while i < a.len() {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}
