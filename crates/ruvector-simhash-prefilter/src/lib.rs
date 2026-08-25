//! # ruvector-simhash-prefilter
//!
//! A *binary sketch prefilter* for approximate-nearest-neighbour candidate
//! reduction. Given a dense float32 vector, we produce a compact bit-signature
//! via **signed random projection** (Charikar's SimHash). At query time we
//! rank candidates by 64/128/256-bit Hamming distance — a fast SIMD-friendly
//! popcount — then rerank the top-M in exact float32 L2.
//!
//! Design goals:
//!
//! * **Trait-based.** `SketchFamily` lets us swap 64/128/256-bit widths and
//!   later plug in learned or orthogonalised projections.
//! * **No unsafe.** Portable Rust; the `u64::count_ones` intrinsic lowers to
//!   `popcntq` on x86_64 and `cnt.8b` on aarch64 without explicit SIMD.
//! * **Real memory math.** A 128-bit sketch is 16 bytes per vector, vs.
//!   `dim * 4` bytes for the raw vector. At `dim=768`, that's a 192× shrink
//!   of the *prefilter* footprint.
//!
//! The prefilter is *not* an index by itself — it is a cheap reducer that
//! composes with HNSW, IVF, brute-force, or any exact reranker.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use thiserror::Error;

/// Errors from prefilter construction and query.
#[derive(Debug, Error)]
pub enum PrefilterError {
    /// Vector dimensionality does not match the family's expected dimension.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Got dimension.
        got: usize,
    },
    /// Query candidate multiplier requested more items than the index holds.
    #[error("candidate multiplier ({m}) exceeds index size ({n})")]
    NotEnoughCandidates {
        /// Requested candidates.
        m: usize,
        /// Available items.
        n: usize,
    },
}

/// A fixed-width binary sketch. `W` is the number of `u64` words (so a
/// `Sketch<2>` is 128 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sketch<const W: usize> {
    /// The packed bit-words.
    pub words: [u64; W],
}

impl<const W: usize> Sketch<W> {
    /// Bit-width of this sketch.
    pub const BITS: usize = W * 64;

    /// Hamming distance to another sketch of the same width.
    #[inline]
    pub fn hamming(&self, other: &Self) -> u32 {
        let mut d = 0u32;
        // Unrolled — the compiler already does this for small W, but being
        // explicit helps auto-vectorisation on some backends.
        for i in 0..W {
            d += (self.words[i] ^ other.words[i]).count_ones();
        }
        d
    }

    /// Zero-initialised sketch (all bits 0).
    pub fn zero() -> Self {
        Self { words: [0u64; W] }
    }
}

/// A projection family that produces `Sketch<W>` values for `dim`-dimensional
/// input vectors. Implementations must be deterministic given the same seed.
pub trait SketchFamily<const W: usize>: Send + Sync {
    /// The expected input dimension.
    fn dim(&self) -> usize;

    /// Encode a raw dense vector into a bit-signature.
    fn sketch(&self, v: &[f32]) -> Result<Sketch<W>, PrefilterError>;

    /// Bytes per stored sketch.
    fn sketch_bytes() -> usize {
        W * 8
    }
}

/// The canonical SimHash / Signed Random Projection family. Draws
/// `W*64` Rademacher (±1) hyperplanes from `StdRng(seed)`. Sign of dot
/// product is the bit.
///
/// We use Rademacher rather than Gaussian projections because (a) storage
/// stays at `dim * W * 64 / 8` bytes with a i8 matrix, and (b) sign-consistency
/// is unchanged under sign-flipped projections — the classic SRP guarantee
/// from Achlioptas '01 still applies.
pub struct SrpFamily<const W: usize> {
    dim: usize,
    /// Row-major `[bits × dim]` Rademacher matrix, packed as i8 for cache
    /// friendliness. We could bit-pack, but i8 keeps the encode kernel a
    /// simple dense mat-vec that the compiler auto-vectorises.
    projections: Vec<i8>,
}

impl<const W: usize> SrpFamily<W> {
    /// Construct a new random SRP family with the given seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        let bits = W * 64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut projections = Vec::with_capacity(bits * dim);
        for _ in 0..(bits * dim) {
            projections.push(if rng.gen_bool(0.5) { 1i8 } else { -1i8 });
        }
        Self { dim, projections }
    }

    /// Bytes the projection matrix occupies in memory.
    pub fn matrix_bytes(&self) -> usize {
        self.projections.len()
    }
}

impl<const W: usize> SketchFamily<W> for SrpFamily<W> {
    fn dim(&self) -> usize {
        self.dim
    }

    fn sketch(&self, v: &[f32]) -> Result<Sketch<W>, PrefilterError> {
        if v.len() != self.dim {
            return Err(PrefilterError::DimensionMismatch {
                expected: self.dim,
                got: v.len(),
            });
        }
        let bits = W * 64;
        let mut out = Sketch::<W>::zero();
        for b in 0..bits {
            let row = &self.projections[b * self.dim..(b + 1) * self.dim];
            // Sum(sign * v[i]) in f32
            let mut acc = 0.0f32;
            for i in 0..self.dim {
                acc += (row[i] as f32) * v[i];
            }
            if acc >= 0.0 {
                let word = b / 64;
                let bit = b % 64;
                out.words[word] |= 1u64 << bit;
            }
        }
        Ok(out)
    }
}

/// A flat brute-force index augmented with a binary prefilter.
///
/// Query flow:
/// 1. Sketch the query.
/// 2. Rank all stored sketches by Hamming distance.
/// 3. Take the top `M = candidate_mult * k` candidates.
/// 4. Rerank those `M` candidates by exact L2 (or squared L2).
///
/// `M > k` trades recall for latency; smaller `M` is faster.
pub struct FlatPrefilterIndex<F: SketchFamily<W>, const W: usize> {
    family: F,
    vectors: Vec<Vec<f32>>,
    sketches: Vec<Sketch<W>>,
}

impl<F: SketchFamily<W>, const W: usize> FlatPrefilterIndex<F, W> {
    /// Build the index from a family and a set of vectors. Encodes eagerly.
    pub fn build(family: F, vectors: Vec<Vec<f32>>) -> Result<Self, PrefilterError> {
        let mut sketches = Vec::with_capacity(vectors.len());
        for v in &vectors {
            sketches.push(family.sketch(v)?);
        }
        Ok(Self {
            family,
            vectors,
            sketches,
        })
    }

    /// Number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Query with the prefilter cascade.
    ///
    /// Returns `k` `(id, distance²)` results.
    pub fn search_prefilter(
        &self,
        query: &[f32],
        k: usize,
        candidate_mult: usize,
    ) -> Result<Vec<(usize, f32)>, PrefilterError> {
        let n = self.vectors.len();
        let m = (k * candidate_mult).min(n);
        if m == 0 {
            return Ok(Vec::new());
        }
        let qs = self.family.sketch(query)?;
        // Score all: (hamming, id)
        let mut scored: Vec<(u32, usize)> = self
            .sketches
            .iter()
            .enumerate()
            .map(|(i, s)| (s.hamming(&qs), i))
            .collect();
        // Partial sort to top-M
        scored.select_nth_unstable_by_key(m - 1, |&(h, _)| h);
        scored.truncate(m);
        // Rerank in exact squared L2
        let mut rerank: Vec<(usize, f32)> = scored
            .into_iter()
            .map(|(_, i)| (i, sq_l2(query, &self.vectors[i])))
            .collect();
        let kk = k.min(rerank.len());
        rerank.select_nth_unstable_by(kk - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        rerank.truncate(kk);
        rerank.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Ok(rerank)
    }

    /// Baseline exact brute-force scan (no prefilter). Used for recall reference.
    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, sq_l2(query, v)))
            .collect();
        let kk = k.min(all.len());
        if kk == 0 {
            return Vec::new();
        }
        all.select_nth_unstable_by(kk - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        all.truncate(kk);
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all
    }

    /// Total prefilter-only memory (bytes) for stored sketches.
    pub fn sketch_footprint_bytes(&self) -> usize {
        self.sketches.len() * F::sketch_bytes()
    }

    /// Total raw vector memory (bytes).
    pub fn raw_footprint_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.len() * 4).sum()
    }
}

/// Squared L2 — no sqrt, monotonic and cheaper.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Compute recall@k of a candidate result set against the ground-truth exact
/// result set. Both slices should be at most length `k`; extra items ignored.
pub fn recall_at_k(candidate: &[(usize, f32)], truth: &[(usize, f32)]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_ids: std::collections::HashSet<usize> = truth.iter().map(|&(i, _)| i).collect();
    let hits = candidate
        .iter()
        .filter(|&&(i, _)| truth_ids.contains(&i))
        .count();
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vecs() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
            vec![0.9, 0.1, 0.0, 0.0], // close to vecs[0]
        ]
    }

    #[test]
    fn sketch_is_deterministic() {
        let f1 = SrpFamily::<2>::new(4, 42);
        let f2 = SrpFamily::<2>::new(4, 42);
        let v = vec![0.5, -0.2, 0.7, 0.1];
        assert_eq!(f1.sketch(&v).unwrap(), f2.sketch(&v).unwrap());
    }

    #[test]
    fn hamming_zero_to_self() {
        let f = SrpFamily::<2>::new(4, 7);
        let v = vec![0.3, 0.4, 0.5, 0.6];
        let s = f.sketch(&v).unwrap();
        assert_eq!(s.hamming(&s), 0);
    }

    #[test]
    fn dim_mismatch_errors() {
        let f = SrpFamily::<1>::new(4, 1);
        assert!(matches!(
            f.sketch(&[0.0, 1.0]),
            Err(PrefilterError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn prefilter_matches_exact_on_tiny_data() {
        let f = SrpFamily::<4>::new(4, 123); // 256-bit sketch
        let idx = FlatPrefilterIndex::build(f, tiny_vecs()).unwrap();
        let q = vec![0.95, 0.05, 0.0, 0.0];
        let exact = idx.search_exact(&q, 1);
        let pre = idx.search_prefilter(&q, 1, 4).unwrap();
        assert_eq!(pre[0].0, exact[0].0);
    }

    #[test]
    fn recall_monotone_in_candidate_mult() {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let normal = rand_distr::Normal::new(0.0f32, 1.0).unwrap();
        let dim = 32;
        let n = 500;
        let vectors: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let queries: Vec<Vec<f32>> = (0..20)
            .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
            .collect();
        let idx = FlatPrefilterIndex::build(SrpFamily::<2>::new(dim, 42), vectors).unwrap();

        let k = 10;
        let mut prev = 0.0;
        for &mult in &[2usize, 5, 10, 25] {
            let mut r_sum = 0.0;
            for q in &queries {
                let truth = idx.search_exact(q, k);
                let pre = idx.search_prefilter(q, k, mult).unwrap();
                r_sum += recall_at_k(&pre, &truth);
            }
            let r = r_sum / queries.len() as f32;
            assert!(
                r + 1e-6 >= prev,
                "recall should be non-decreasing in candidate multiplier: {r} < {prev}"
            );
            prev = r;
        }
        // With mult=25 on n=500 we scan 50% of the corpus — recall must be perfect.
        assert!(prev >= 0.999, "recall at mult=25 was {prev}");
    }
}
