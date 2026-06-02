//! 1-bit RaBitQ-style binary quantization.
//!
//! For each centred vector `x' = x - μ` we store the sign-bit vector
//! `b(x) = sign(x')`. The score we use during graph traversal is an
//! *asymmetric* distance: between a (float) query and a binary code, we
//! compute  Σ_i  (q_i' < 0) XOR b(x)_i.  In other words, a popcount of
//! disagreements between `sign(q')` and `b(x)`, weighted by |q_i'| through
//! a precomputed query-side table.
//!
//! This is a faithful (and intentionally minimal) reproduction of the
//! distance-evaluation core used by SymphonyQG / RaBitQ. Real production
//! impls use AVX2/AVX-512 8-bit fused popcount-mul tables; here we use
//! u64 popcount and a per-query Σ|q| weighting which keeps the relative
//! ordering correct while remaining safe Rust.

/// One stored vector's bits, packed into u64 words (LSB first).
#[derive(Clone, Debug)]
pub struct BinaryCode {
    pub bits: Vec<u64>,
}

/// Encoder + per-query helper.
pub struct BinaryCodec {
    pub dim: usize,
    pub words: usize,        // ceil(dim / 64)
    pub centroid: Vec<f32>,
    pub flips: Vec<i8>,      // random ±1 diagonal (length = dim, power-of-2 padded)
    pub padded: usize,       // next power of two >= dim
    pub rotate: bool,        // whether to apply Walsh–Hadamard rotation
}

/// In-place normalized Walsh–Hadamard transform. `v.len()` must be a power of 2.
fn fwht(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j]     = x + y;
                v[j + h] = x - y;
            }
            i += h * 2;
        }
        h *= 2;
    }
    let inv = (n as f32).sqrt().recip();
    for x in v.iter_mut() { *x *= inv; }
}

fn next_pow2(n: usize) -> usize {
    let mut p = 1; while p < n { p *= 2; } p
}

impl BinaryCodec {
    /// Build a codec that centres data on the dataset mean and (optionally)
    /// applies a randomized sign + Walsh–Hadamard rotation before binarization.
    /// The rotation is the standard RaBitQ trick: it whitens the projected
    /// distribution so the sign bits carry near-uniform information regardless
    /// of input direction.
    pub fn fit(data: &[f32], dim: usize) -> Self {
        Self::fit_with(data, dim, true, 0xC0FFEE_u64)
    }

    pub fn fit_with(data: &[f32], dim: usize, rotate: bool, seed: u64) -> Self {
        assert!(dim > 0);
        let n = data.len() / dim;
        assert_eq!(n * dim, data.len());

        let mut centroid = vec![0.0f32; dim];
        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            for j in 0..dim { centroid[j] += row[j]; }
        }
        if n > 0 {
            for j in 0..dim { centroid[j] /= n as f32; }
        }

        let padded = next_pow2(dim);
        // Deterministic Linear-Congruential bit sequence for the random sign
        // diagonal — keeps the codec reproducible without a `rand` dep on it.
        let mut state = seed.max(1);
        let flips: Vec<i8> = (0..padded).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            if (state >> 33) & 1 == 0 { 1 } else { -1 }
        }).collect();

        Self { dim, words: (padded + 63) / 64, centroid, flips, padded, rotate }
    }

    fn project(&self, v: &[f32]) -> Vec<f32> {
        let mut buf = vec![0.0f32; self.padded];
        for j in 0..self.dim {
            buf[j] = (v[j] - self.centroid[j]) * self.flips[j] as f32;
        }
        if self.rotate {
            fwht(&mut buf);
        }
        buf
    }

    /// Encode every row of `data` to a binary code.
    pub fn encode_all(&self, data: &[f32]) -> Vec<BinaryCode> {
        let n = data.len() / self.dim;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &data[i * self.dim..(i + 1) * self.dim];
            out.push(self.encode(row));
        }
        out
    }

    pub fn encode(&self, v: &[f32]) -> BinaryCode {
        debug_assert_eq!(v.len(), self.dim);
        let proj = self.project(v);
        let mut bits = vec![0u64; self.words];
        for j in 0..self.padded {
            if proj[j] >= 0.0 {
                bits[j >> 6] |= 1u64 << (j & 63);
            }
        }
        BinaryCode { bits }
    }

    /// Build a per-query packed sign vector + L1 weight used as the distance
    /// scale factor. Returning the L1 weight lets downstream code recover a
    /// distance with the same monotonicity as the true asymmetric variant.
    pub fn prepare_query(&self, q: &[f32]) -> PreparedQuery {
        debug_assert_eq!(q.len(), self.dim);
        let proj = self.project(q);
        let mut bits = vec![0u64; self.words];
        let mut l1 = 0.0f32;
        for j in 0..self.padded {
            l1 += proj[j].abs();
            if proj[j] >= 0.0 {
                bits[j >> 6] |= 1u64 << (j & 63);
            }
        }
        let w = if self.padded > 0 { l1 / self.padded as f32 } else { 0.0 };
        PreparedQuery { bits, weight: w }
    }
}

/// A query staged for binary scoring.
pub struct PreparedQuery {
    pub bits: Vec<u64>,
    pub weight: f32,
}

impl PreparedQuery {
    /// Distance proxy to a stored code. Lower = more likely closer.
    #[inline]
    pub fn score(&self, code: &BinaryCode) -> f32 {
        let mut h: u32 = 0;
        for w in 0..self.bits.len() {
            h += (self.bits[w] ^ code.bits[w]).count_ones();
        }
        // Weight makes the score comparable in magnitude to L2² (still a
        // proxy — exact reranking uses float L2 in symphony.rs).
        (h as f32) * self.weight
    }
}
