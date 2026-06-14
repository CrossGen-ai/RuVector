//! Distance functions for HCNNG.
//!
//! Trait-based so backends (L2, IP, cosine) can be swapped. Default is L2^2
//! because monotonic preservation is all the graph/MST construction needs and
//! it skips a sqrt per call.

use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// Squared L2. Monotone-equivalent to L2 for ordering.
    L2Sq,
    /// Negated inner product (so smaller = closer, like L2).
    NegIP,
    /// Cosine distance: 1 - cos. Inputs need not be normalized; we normalize on the fly.
    Cosine,
}

pub trait Distance: Send + Sync {
    fn d(&self, a: &[f32], b: &[f32]) -> f32;
}

pub struct L2Sq;
impl Distance for L2Sq {
    #[inline]
    fn d(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut s = 0.0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }
}

pub struct NegIP;
impl Distance for NegIP {
    #[inline]
    fn d(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut s = 0.0f32;
        for i in 0..a.len() {
            s += a[i] * b[i];
        }
        -s
    }
}

pub struct Cosine;
impl Distance for Cosine {
    #[inline]
    fn d(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for i in 0..a.len() {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
        1.0 - dot / denom
    }
}

pub fn make(metric: Metric) -> Box<dyn Distance> {
    match metric {
        Metric::L2Sq => Box::new(L2Sq),
        Metric::NegIP => Box::new(NegIP),
        Metric::Cosine => Box::new(Cosine),
    }
}
