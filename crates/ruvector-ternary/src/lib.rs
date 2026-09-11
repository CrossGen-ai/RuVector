//! ruvector-ternary
//!
//! Ternary {-1, 0, +1} vector encoding for high-recall ANN prefilter.
//!
//! Standard 1-bit binary quantization (sign of each coordinate) is popular
//! because Hamming distance is a single `popcount(a ^ b)` per 64-D chunk. It
//! loses one degree of freedom the query most cares about: coordinates whose
//! magnitude is small are the ones whose sign is *least* informative and yet
//! contribute equally to Hamming distance. Ternary encoding introduces a
//! per-vector threshold `theta` and maps
//!
//! ```text
//!   x[i] ->  +1  if x[i] >  theta
//!            -1  if x[i] < -theta
//!             0  otherwise
//! ```
//!
//! Small-magnitude coordinates land in the "0" bucket and are excluded from
//! the sign comparison — recovering some of the information a plain binary
//! encoding throws away.
//!
//! We store each vector as two bitplanes:
//!
//! * `sign`  — 1 if coordinate is `+1`, 0 if `-1` or `0`
//! * `mask`  — 1 if coordinate is *non-zero* (i.e. `|x[i]| > theta`)
//!
//! With that layout the ternary inner-product-like distance
//!
//! ```text
//!   T_dist(a, b) = sum_i [a[i] != 0 AND b[i] != 0 AND a[i] != b[i]]
//! ```
//!
//! reduces to a single fused expression per 64-bit chunk:
//!
//! ```text
//!   popcount( (sign_a ^ sign_b) & mask_a & mask_b )
//! ```
//!
//! Two `xor`s and two `and`s plus a `popcnt` — no floats, cache-friendly,
//! auto-vectorizes on x86 (`popcnt`) and aarch64 (`cnt` + `addv`).
//!
//! # Design goals
//!
//! * **Trait-swappable backends**: [`Encoder`] and [`Distance`] traits so
//!   binary / ternary / int8 alternatives all present the same shape to a
//!   recall harness.
//! * **Real memory math**: [`Encoded::bytes()`] returns the honest storage
//!   footprint per vector, exposed to the benchmark so the research doc can
//!   report bytes-per-vector alongside recall.
//! * **Deterministic**: seeded RNG throughout, so the numbers in the paper
//!   reproduce exactly.

pub mod binary;
pub mod int8;
pub mod ternary;

/// One dense fp32 vector — the ground-truth representation used to compute
/// exact top-k for recall measurement.
pub type Vector = Vec<f32>;

/// A quantized-vector distance function. Distances are unsigned integers so
/// that ternary/binary/int8 backends all sort naturally with no float
/// comparison surprises. Smaller = more similar (Hamming-like semantics).
pub trait Distance {
    type Code;
    /// Distance between two encoded codes. Must be symmetric and admit a
    /// stable total order for tie-broken argmin.
    fn dist(&self, a: &Self::Code, b: &Self::Code) -> u32;
}

/// An encoder that maps an fp32 vector into a compact code plus per-vector
/// metadata. The encoder captures whatever calibration state (thresholds,
/// scales) is needed to reproduce the code deterministically.
pub trait Encoder {
    type Code: Clone;

    /// Encode one vector. `dim` is captured at construction and enforced.
    fn encode(&self, v: &[f32]) -> Self::Code;

    /// Honest bytes-per-code, used for memory-vs-recall reporting.
    fn bytes_per_code(&self) -> usize;

    /// Short human-readable name for tables ("binary", "ternary", "int8").
    fn name(&self) -> &'static str;
}

/// Exact fp32 L2 for oracle top-k.
#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Deterministic StdRng for reproducibility.
pub fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed)
}

/// Number of 64-bit words required to hold `dim` bits.
#[inline]
pub const fn words_for(dim: usize) -> usize {
    (dim + 63) / 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_for_math() {
        assert_eq!(words_for(0), 0);
        assert_eq!(words_for(1), 1);
        assert_eq!(words_for(64), 1);
        assert_eq!(words_for(65), 2);
        assert_eq!(words_for(128), 2);
    }

    #[test]
    fn l2_positive_and_zero_on_self() {
        let a = vec![0.0, 1.0, 2.0];
        let b = vec![1.0, 1.0, 2.0];
        assert_eq!(l2(&a, &a), 0.0);
        assert!((l2(&a, &b) - 1.0).abs() < 1e-6);
    }
}
