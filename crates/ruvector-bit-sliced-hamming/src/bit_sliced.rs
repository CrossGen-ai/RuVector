//! Bit-transposed storage with dim-band early termination.
//!
//! # Layout
//!
//! Row-major storage lays out `n * (dim/64)` u64 words, one row per vector.
//! Bit-transposed (bit-sliced) storage lays out `dim` bit-slices, each
//! `chunks = ceil(n / 64)` u64 words long. Word
//! `slice_words[bit * chunks + c]` holds, in bit `k`, the value of
//! dimension `bit` of vector `c * 64 + k`.
//!
//! # Traversal
//!
//! For a single query we iterate over `dim` bit-slices (instead of `n`
//! rows). Each slice contributes a mismatch bit for every currently-active
//! vector; a `u8` counter per vector accumulates its Hamming distance so
//! far. Between bands of [`BAND`] dims we sweep the active bitmap:
//!
//! * Lower bound `L_j = counters[j]` (only grows over time).
//! * Upper bound `U_j = counters[j] + (dim - dims_done)`.
//! * `U_k` = k-th smallest `U_j` across the *active* set is an upper bound
//!   on the k-th best final distance.
//! * Any `j` with `L_j > U_k` cannot be in the final top-k → prune.
//!
//! At `dims_done == dim`, `L_j == U_j` is the true distance for every
//! surviving `j`; the top-k is a straight selection over the active set.
//! The pruning is provably lossless, so the top-k matches the naive
//! backend exactly (verified by cross-backend test).
//!
//! # When does this actually win?
//!
//! Not on iid-random binary data — distances cluster tightly around dim/2
//! and pruning is weak. Bit-sliced wins on structured (clustered / learned
//! embedding) data where the query is close to a small subset of the
//! corpus, and it wins for small `k` and moderate `dim` where the per-
//! set-bit `trailing_zeros` inner loop stays cheap. The
//! `src/bin/benchmark.rs` binary reports both regimes so the crossover
//! is transparent.

use crate::{finalize_topk, HammingTopK, PackedCorpus, Query, TopK, VectorId};

/// Number of dimensions between candidate-set pruning sweeps.
pub const BAND: usize = 32;

pub struct BitSlicedEarlyTerm {
    n: usize,
    dim: usize,
    /// bit-transposed corpus: `dim * chunks` u64 words.
    slice_words: Vec<u64>,
    chunks: usize,
}

impl HammingTopK for BitSlicedEarlyTerm {
    fn build(corpus: PackedCorpus) -> Self {
        assert!(corpus.dim <= 255, "BitSlicedEarlyTerm supports dim<=255 (u8 counters)");
        let n = corpus.n;
        let dim = corpus.dim;
        let chunks = n.div_ceil(64).max(1);
        let mut slice_words = vec![0u64; dim * chunks];
        let w = dim / 64;
        for i in 0..n {
            let base = i * w;
            let chunk = i / 64;
            let lane = (i % 64) as u32;
            for bit in 0..dim {
                let word = corpus.words[base + bit / 64];
                let b = (word >> (bit % 64)) & 1;
                if b == 1 {
                    slice_words[bit * chunks + chunk] |= 1u64 << lane;
                }
            }
        }
        Self { n, dim, slice_words, chunks }
    }

    fn topk(&self, query: Query<'_>, k: usize) -> TopK {
        let n = self.n;
        if n == 0 || k == 0 { return Vec::new(); }
        let k = k.min(n);
        let dim = self.dim;
        let chunks = self.chunks;
        let padded_n = chunks * 64;

        let mut counters: Vec<u8> = vec![0u8; padded_n];
        let mut active: Vec<u64> = vec![!0u64; chunks];
        let tail = n % 64;
        if tail != 0 {
            active[chunks - 1] = (1u64 << tail) - 1;
        }

        let mut dims_done = 0usize;
        while dims_done < dim {
            let band_end = (dims_done + BAND).min(dim);
            for bit in dims_done..band_end {
                let qb = (query[bit / 64] >> (bit % 64)) & 1;
                let qmask = if qb == 1 { !0u64 } else { 0u64 };
                let slice_base = bit * chunks;
                for c in 0..chunks {
                    let a = active[c];
                    if a == 0 { continue; }
                    let mut m = (self.slice_words[slice_base + c] ^ qmask) & a;
                    while m != 0 {
                        let lane = m.trailing_zeros() as usize;
                        counters[c * 64 + lane] =
                            counters[c * 64 + lane].saturating_add(1);
                        m &= m - 1;
                    }
                }
            }
            dims_done = band_end;

            if dims_done < dim {
                // Compute U_k = k-th smallest upper-bound over actives.
                // Upper bound U_j = counters[j] + (dim - dims_done).
                // Equivalently we can just compute the k-th smallest
                // counters[j] and add (dim - dims_done) once.
                let mut alive_counters: Vec<u16> = Vec::with_capacity(padded_n);
                let mut alive_ids: Vec<u32> = Vec::with_capacity(padded_n);
                for c in 0..chunks {
                    let mut mask = active[c];
                    while mask != 0 {
                        let lane = mask.trailing_zeros() as usize;
                        let idx = c * 64 + lane;
                        alive_counters.push(counters[idx] as u16);
                        alive_ids.push(idx as u32);
                        mask &= mask - 1;
                    }
                }
                if alive_counters.len() > k {
                    let mut sorted = alive_counters.clone();
                    let (_, kth, _) = sorted.select_nth_unstable(k - 1);
                    let u_k = (*kth) as u32 + (dim - dims_done) as u32;
                    // Prune j whose lower bound (counters[j]) > u_k.
                    for c in 0..chunks {
                        let a = active[c];
                        if a == 0 { continue; }
                        let mut alive = a;
                        let mut mask = a;
                        while mask != 0 {
                            let lane = mask.trailing_zeros() as usize;
                            let idx = c * 64 + lane;
                            if (counters[idx] as u32) > u_k {
                                alive &= !(1u64 << lane);
                            }
                            mask &= mask - 1;
                        }
                        active[c] = alive;
                    }
                }
            }
        }

        // Final selection over surviving actives.
        let mut heap: std::collections::BinaryHeap<(u8, u32)> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        for c in 0..chunks {
            let mut m = active[c];
            while m != 0 {
                let lane = m.trailing_zeros() as usize;
                let idx = c * 64 + lane;
                let d = counters[idx];
                if heap.len() < k {
                    heap.push((d, idx as u32));
                } else {
                    let top = *heap.peek().unwrap();
                    let cand = (d, idx as u32);
                    if cand.0 < top.0 || (cand.0 == top.0 && cand.1 < top.1) {
                        heap.pop();
                        heap.push(cand);
                    }
                }
                m &= m - 1;
            }
        }
        let out: Vec<(VectorId, u32)> = heap.into_iter().map(|(d, i)| (i, d as u32)).collect();
        finalize_topk(out)
    }

    fn name(&self) -> &'static str { "BitSlicedEarlyTerm" }

    fn storage_bytes(&self) -> usize { self.slice_words.len() * 8 }
}
