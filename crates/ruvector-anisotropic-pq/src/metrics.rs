//! Tiny vector helpers used by every backend.

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[inline]
pub fn norm2(a: &[f32]) -> f32 {
    let mut s = 0f32;
    for &v in a {
        s += v * v;
    }
    s
}

pub fn normalize_in_place(a: &mut [f32]) {
    let n = norm2(a).sqrt();
    if n > 1e-12 {
        let inv = 1.0 / n;
        for v in a.iter_mut() {
            *v *= inv;
        }
    }
}
