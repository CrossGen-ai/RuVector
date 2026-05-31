//! Distance metrics for DEG. Trait-based so backends can swap in
//! quantised distances (e.g. RaBitQ, LVQ) without changing the graph code.

use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    L2Sq,
    Cosine,
}

impl Metric {
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Metric::L2Sq => l2_sq(a, b),
            Metric::Cosine => cosine_dist(a, b),
        }
    }
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    // 4-way unroll; the autovectoriser turns this into SSE/NEON in release.
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

#[inline]
pub fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na * nb).sqrt().max(1e-12);
    1.0 - dot / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_matches_naive() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 0.0, 1.0, 4.0, 7.0];
        let naive: f32 = a.iter().zip(&b).map(|(x, y)| { let d: f32 = x - y; d * d }).sum();
        assert!((l2_sq(&a, &b) - naive).abs() < 1e-5);
    }

    #[test]
    fn cosine_self_is_zero() {
        let a = vec![0.3, -0.5, 1.2, 0.8];
        assert!(cosine_dist(&a, &a).abs() < 1e-5);
    }
}
