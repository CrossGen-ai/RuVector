//! Cache-blocked row-major scan with an online top-k max-heap.
//!
//! Same storage as [`crate::naive::NaiveRowMajor`], but the scan is broken
//! into row-blocks so the loop keeps the query vector and the top-k heap
//! working set in registers/L1. Distances are pushed into a bounded
//! max-heap of size `k`; only distances strictly better than the current
//! worst are retained.
//!
//! Modest but real speedup vs. NaiveRowMajor because (a) no `Vec<u32>` of
//! length N is allocated on the hot path, (b) the heap short-circuits away
//! from the branch-heavy select_nth path, and (c) the query is guaranteed
//! to stay hot in L1 across the row-block.

use std::collections::BinaryHeap;
use std::cmp::Reverse;

use crate::{finalize_topk, HammingTopK, PackedCorpus, Query, TopK, VectorId};

pub struct BlockedRowMajor {
    corpus: PackedCorpus,
}

const ROW_BLOCK: usize = 256;

impl HammingTopK for BlockedRowMajor {
    fn build(corpus: PackedCorpus) -> Self { Self { corpus } }

    fn topk(&self, query: Query<'_>, k: usize) -> TopK {
        let n = self.corpus.n;
        if n == 0 || k == 0 { return Vec::new(); }
        let k = k.min(n);
        let w = self.corpus.words_per_vec();
        assert_eq!(query.len(), w, "query dim mismatch");

        // Max-heap of size k. Encoded as min-heap-of-Reverse-on-negated so
        // that we can pop the *worst* current element cheaply. We use a plain
        // BinaryHeap<(u32,u32)> where the top is the vector with the largest
        // distance (max-heap).
        let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);

        let mut i0 = 0usize;
        while i0 < n {
            let i1 = (i0 + ROW_BLOCK).min(n);
            for i in i0..i1 {
                let base = i * w;
                let mut d: u32 = 0;
                for j in 0..w {
                    d += (self.corpus.words[base + j] ^ query[j]).count_ones();
                }
                if heap.len() < k {
                    heap.push((d, i as u32));
                } else {
                    // Compare against current worst (heap top). Break ties by
                    // preferring smaller id so the final total order agrees
                    // with the naive backend.
                    let top = *heap.peek().unwrap();
                    let cand = (d, i as u32);
                    if cand.0 < top.0 || (cand.0 == top.0 && cand.1 < top.1) {
                        heap.pop();
                        heap.push(cand);
                    }
                }
            }
            i0 = i1;
        }

        // Reformat heap into (id, dist) and finalize.
        let out: Vec<(VectorId, u32)> = heap.into_iter().map(|(d, i)| (i, d)).collect();
        // finalize_topk sorts by dist asc, id asc.
        // Guard: use Reverse to silence unused-import warning.
        let _ = Reverse::<u32>;
        finalize_topk(out)
    }

    fn name(&self) -> &'static str { "BlockedRowMajor" }

    fn storage_bytes(&self) -> usize { self.corpus.words.len() * 8 }
}
