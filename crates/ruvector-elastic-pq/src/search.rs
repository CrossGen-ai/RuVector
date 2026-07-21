//! Top-k asymmetric-distance-computation (ADC) search over an EBA-PQ
//! code array. Straight-line scalar loop — no SIMD, no rerank stage —
//! kept intentionally simple so the benchmark numbers isolate the effect
//! of the bit allocation strategy rather than the scan implementation.

use crate::pq::{ElasticPq, ElasticPqError};

/// A single search hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// Index of the vector in the encoded database.
    pub index: u32,
    /// Approximate squared L2 distance from the query.
    pub distance: f32,
}

/// Simple top-k ADC searcher over a packed code array.
pub struct AdcSearcher<'a> {
    pq: &'a ElasticPq,
    codes: &'a [u8],
    n: usize,
}

impl<'a> AdcSearcher<'a> {
    /// Wrap a trained model + a code array. `codes.len()` must equal
    /// `n * pq.code_bytes()`.
    pub fn new(pq: &'a ElasticPq, codes: &'a [u8], n: usize) -> Self {
        assert_eq!(codes.len(), n * pq.code_bytes());
        AdcSearcher { pq, codes, n }
    }

    /// Return the `k` closest DB entries to `query` under ADC.
    pub fn topk(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, ElasticPqError> {
        let tables = self.pq.adc_tables(query)?;
        let m = self.pq.m();
        let mut all: Vec<SearchResult> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let code = &self.codes[i * m..(i + 1) * m];
            let mut acc = 0f32;
            for j in 0..m {
                acc += tables[j][code[j] as usize];
            }
            all.push(SearchResult {
                index: i as u32,
                distance: acc,
            });
        }
        all.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        all.truncate(k);
        Ok(all)
    }
}

/// Recall@k: fraction of ground-truth top-k indices that appear in the
/// approximate top-k. `truth` and `approx` are both length-k slices of
/// vector ids.
pub fn recall_at_k(truth: &[u32], approx: &[u32]) -> f32 {
    let mut hits = 0;
    for a in approx {
        if truth.contains(a) {
            hits += 1;
        }
    }
    hits as f32 / truth.len().max(1) as f32
}
