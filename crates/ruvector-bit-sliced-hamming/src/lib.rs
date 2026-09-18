//! Bit-sliced (bit-transposed) storage for Hamming top-K.
//!
//! Three swappable backends implement the [`HammingTopK`] trait:
//!
//! * [`naive::NaiveRowMajor`] — dense row-major storage, POPCNT per vector.
//! * [`blocked::BlockedRowMajor`] — cache-blocked row-major with online top-K heap.
//! * [`bit_sliced::BitSlicedEarlyTerm`] — bit-transposed layout with candidate
//!   bitmap and early-termination pruning by dimension groups.
//!
//! All backends operate on **fixed-dimension binary vectors** where `dim`
//! must be a multiple of 64. Storage is exposed as a builder that indexes a
//! corpus once; the backend is chosen at construction.
//!
//! Correctness invariant enforced by tests: for identical `(corpus, query, k)`,
//! all three backends return the same *set* of top-k vector ids (ties broken
//! deterministically by id ascending). See `tests/` and the built-in
//! `benchmark` binary.

pub mod naive;
pub mod blocked;
pub mod bit_sliced;

/// Dimension of a binary vector, expressed in bits. Must be a multiple of 64.
pub type DimBits = usize;

/// Vector identifier in the corpus (0-based).
pub type VectorId = u32;

/// A top-k result: `(vector_id, hamming_distance)`, sorted ascending by
/// distance, then ascending by id.
pub type TopK = Vec<(VectorId, u32)>;

/// Row-major packed binary corpus: `n` vectors of `dim` bits, stored as
/// `n * (dim/64)` u64 words. Vector `i` occupies words `i*words_per_vec ..
/// (i+1)*words_per_vec`.
#[derive(Clone, Debug)]
pub struct PackedCorpus {
    pub n: usize,
    pub dim: DimBits,
    pub words: Vec<u64>,
}

impl PackedCorpus {
    pub fn new(n: usize, dim: DimBits) -> Self {
        assert!(dim % 64 == 0, "dim must be a multiple of 64");
        Self { n, dim, words: vec![0u64; n * (dim / 64)] }
    }

    pub fn words_per_vec(&self) -> usize { self.dim / 64 }

    pub fn set_vector(&mut self, id: usize, packed: &[u64]) {
        let w = self.words_per_vec();
        assert_eq!(packed.len(), w);
        self.words[id * w..(id + 1) * w].copy_from_slice(packed);
    }

    pub fn vec_words(&self, id: usize) -> &[u64] {
        let w = self.words_per_vec();
        &self.words[id * w..(id + 1) * w]
    }
}

/// Query is a single packed binary vector: `dim/64` u64 words.
pub type Query<'a> = &'a [u64];

/// Trait implemented by each backend.
pub trait HammingTopK {
    /// Build an index from a packed corpus. Backends may reformat storage
    /// (e.g. bit-transpose); the input `corpus` is consumed or borrowed
    /// depending on backend.
    fn build(corpus: PackedCorpus) -> Self where Self: Sized;

    /// Return the top-`k` nearest neighbours of `query` by Hamming distance.
    fn topk(&self, query: Query<'_>, k: usize) -> TopK;

    /// Human-readable backend name for reports.
    fn name(&self) -> &'static str;

    /// Bytes of *searchable* storage (excludes scratch buffers).
    fn storage_bytes(&self) -> usize;
}

/// Deterministic tie-breaking sort: ascending by distance then id.
pub(crate) fn finalize_topk(mut heap: Vec<(u32, u32)>) -> TopK {
    heap.sort_unstable_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    heap
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn random_corpus(n: usize, dim: DimBits, seed: u64) -> PackedCorpus {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut c = PackedCorpus::new(n, dim);
        let w = c.words_per_vec();
        for i in 0..n {
            let mut buf = vec![0u64; w];
            for word in &mut buf { *word = rng.gen::<u64>(); }
            c.set_vector(i, &buf);
        }
        c
    }

    fn random_query(dim: DimBits, seed: u64) -> Vec<u64> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..dim/64).map(|_| rng.gen::<u64>()).collect()
    }

    #[test]
    fn backends_agree_on_topk_ids() {
        let c = random_corpus(2000, 128, 42);
        let q = random_query(128, 7);
        let k = 10;

        let a = naive::NaiveRowMajor::build(c.clone()).topk(&q, k);
        let b = blocked::BlockedRowMajor::build(c.clone()).topk(&q, k);
        let d = bit_sliced::BitSlicedEarlyTerm::build(c.clone()).topk(&q, k);

        // Distances must match exactly (order can differ only on ties, but
        // finalize_topk enforces a total order).
        assert_eq!(a, b, "naive vs blocked disagree");
        assert_eq!(a, d, "naive vs bit-sliced disagree");
    }

    #[test]
    fn hamming_distance_is_correct_on_known_pairs() {
        // dim = 64, vector 0 = all zeros, vector 1 = 0xFF (8 ones), vector 2 = !0 (64 ones).
        let mut c = PackedCorpus::new(3, 64);
        c.set_vector(0, &[0u64]);
        c.set_vector(1, &[0xFFu64]);
        c.set_vector(2, &[!0u64]);

        let q = [0u64];
        let want = vec![(0u32, 0u32), (1u32, 8u32), (2u32, 64u32)];

        assert_eq!(naive::NaiveRowMajor::build(c.clone()).topk(&q, 3), want);
        assert_eq!(blocked::BlockedRowMajor::build(c.clone()).topk(&q, 3), want);
        assert_eq!(bit_sliced::BitSlicedEarlyTerm::build(c.clone()).topk(&q, 3), want);
    }

    #[test]
    fn empty_topk_is_empty() {
        let c = PackedCorpus::new(0, 64);
        let q = [0u64];
        assert!(naive::NaiveRowMajor::build(c.clone()).topk(&q, 10).is_empty());
    }

    #[test]
    fn dim_multiple_of_64_enforced() {
        let result = std::panic::catch_unwind(|| PackedCorpus::new(1, 65));
        assert!(result.is_err());
    }
}
