//! Baseline row-major popcount scan.
//!
//! For each of `n` vectors, XORs `dim/64` u64 words against the query and
//! accumulates `u64::count_ones()`. Distances are collected into an owned
//! `Vec<u32>` and then partial-sorted for top-k. No blocking, no heap; the
//! simplest possible implementation, useful as a correctness oracle and as
//! the "beat me" baseline in benchmarks.

use crate::{finalize_topk, HammingTopK, PackedCorpus, Query, TopK, VectorId};

pub struct NaiveRowMajor {
    corpus: PackedCorpus,
}

impl HammingTopK for NaiveRowMajor {
    fn build(corpus: PackedCorpus) -> Self { Self { corpus } }

    fn topk(&self, query: Query<'_>, k: usize) -> TopK {
        let n = self.corpus.n;
        if n == 0 || k == 0 { return Vec::new(); }
        let k = k.min(n);
        let w = self.corpus.words_per_vec();
        assert_eq!(query.len(), w, "query dim mismatch");

        let mut dists: Vec<(VectorId, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * w;
            let mut d: u32 = 0;
            for j in 0..w {
                d += (self.corpus.words[base + j] ^ query[j]).count_ones();
            }
            dists.push((i as u32, d));
        }

        // partial sort for top-k
        dists.select_nth_unstable_by(k - 1, |a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        dists.truncate(k);
        finalize_topk(dists)
    }

    fn name(&self) -> &'static str { "NaiveRowMajor" }

    fn storage_bytes(&self) -> usize { self.corpus.words.len() * 8 }
}
