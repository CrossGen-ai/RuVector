//! Locally-adaptive Vector Quantization (LVQ-8).
//!
//! Per vector `v ∈ R^r`, store
//! ```text
//!   lo   = min_j v[j]
//!   step = (max_j v[j] − lo) / 255
//!   code[j] = round((v[j] − lo) / step)    ∈ {0, …, 255}
//! ```
//! Decoded value is `lo + step · code[j]`. The encoded form costs `r + 8`
//! bytes per vector (`r` u8 codes + two f32 scalars) which is roughly 4× more
//! compact than f32 once `r ≥ 16` or so.
//!
//! For asymmetric L2 distance, queries stay in float and database vectors are
//! decoded on the fly:
//!
//! ```text
//!   ||q − v||² = Σ (q_j − lo − step · code_j)²
//!              = Σ ((q_j − lo) − step · code_j)²
//! ```
//!
//! Promoting the `(q_j − lo)` shift into a once-per-vector setup lets the inner
//! loop run as plain f32 ops over `q'_j = q_j − lo` and `step · code_j`,
//! identical to FAISS's `IndexLVQ8` reference implementation.

/// A single LVQ-8 quantised vector.
#[derive(Clone, Debug)]
pub struct LvqCode {
    /// 8-bit per-dimension codes.
    pub codes: Vec<u8>,
    /// Per-vector additive offset.
    pub lo: f32,
    /// Per-vector quantisation step (range / 255).
    pub step: f32,
}

impl LvqCode {
    /// Encode a single vector.
    pub fn encode(v: &[f32]) -> Self {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &x in v {
            if x < lo {
                lo = x;
            }
            if x > hi {
                hi = x;
            }
        }
        if !lo.is_finite() {
            lo = 0.0;
        }
        if !hi.is_finite() {
            hi = 0.0;
        }
        let range = (hi - lo).max(1e-12);
        let step = range / 255.0;
        let inv_step = 1.0 / step;
        let mut codes = Vec::with_capacity(v.len());
        for &x in v {
            let c = ((x - lo) * inv_step).round();
            let c = c.clamp(0.0, 255.0) as u8;
            codes.push(c);
        }
        Self { codes, lo, step }
    }

    /// Decoded value for a single dimension.
    #[inline]
    pub fn decode_one(&self, j: usize) -> f32 {
        self.lo + self.step * (self.codes[j] as f32)
    }

    /// Materialise the full decoded vector.
    pub fn decode(&self) -> Vec<f32> {
        self.codes.iter().map(|&c| self.lo + self.step * c as f32).collect()
    }

    /// Storage cost in bytes (codes + two f32 scalars).
    pub fn bytes(&self) -> usize {
        self.codes.len() + 8
    }

    /// Asymmetric squared L2: query is float, database vector is this code.
    ///
    /// We rewrite each term as `(q_j − lo − step·c_j)²` with no intermediate
    /// `Vec` allocation; the auto-vectoriser turns this into a tight loop for
    /// the dimensions LeanVec produces (typically 32–128).
    #[inline]
    pub fn asym_l2_sq(&self, q: &[f32]) -> f32 {
        debug_assert_eq!(q.len(), self.codes.len());
        let mut s = 0.0_f32;
        for j in 0..self.codes.len() {
            let d = q[j] - self.lo - self.step * (self.codes[j] as f32);
            s += d * d;
        }
        s
    }
}

/// A bag of LVQ-8 codes plus a shared dimensionality. Owns the codes — keep
/// the original f32 vectors elsewhere if you need exact rerank.
#[derive(Clone, Debug, Default)]
pub struct LvqCodebook {
    /// Dimensionality of the (possibly projected) vectors being encoded.
    pub dim: usize,
    /// One code per vector, in insertion order.
    pub codes: Vec<LvqCode>,
}

impl LvqCodebook {
    /// Create an empty codebook for a given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self { dim, codes: Vec::new() }
    }

    /// Encode and append a vector.
    pub fn push(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.dim);
        self.codes.push(LvqCode::encode(v));
    }

    /// Number of encoded vectors.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the codebook is empty.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Total bytes occupied by all codes.
    pub fn bytes(&self) -> usize {
        self.codes.iter().map(|c| c.bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip_under_step() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1 - 3.2).collect();
        let c = LvqCode::encode(&v);
        let d = c.decode();
        // Each component must agree to within one quantisation step.
        for j in 0..v.len() {
            assert!((v[j] - d[j]).abs() <= c.step + 1e-6,
                "j={j}: v={} decoded={} step={}", v[j], d[j], c.step);
        }
    }

    #[test]
    fn asym_l2_matches_decoded_l2() {
        let v: Vec<f32> = (0..32).map(|i| ((i * 13) as f32).sin()).collect();
        let q: Vec<f32> = (0..32).map(|i| ((i * 7) as f32).cos()).collect();
        let c = LvqCode::encode(&v);
        let decoded = c.decode();
        let mut expected = 0.0;
        for j in 0..32 {
            let d = q[j] - decoded[j];
            expected += d * d;
        }
        let got = c.asym_l2_sq(&q);
        assert!((got - expected).abs() < 1e-4, "got {got} expected {expected}");
    }

    #[test]
    fn codebook_byte_accounting() {
        let mut cb = LvqCodebook::new(16);
        for i in 0..10 {
            let v: Vec<f32> = (0..16).map(|j| (i + j) as f32 * 0.01).collect();
            cb.push(&v);
        }
        // 16 codes + 8 scalar bytes per vector, times 10 vectors.
        assert_eq!(cb.bytes(), 10 * (16 + 8));
    }
}
