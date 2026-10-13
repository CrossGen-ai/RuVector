//! Trait that all four indexes implement so the demo and bench can swap them.

#[derive(Clone, Copy, Debug)]
pub struct SearchResult {
    pub id: u32,
    pub score: f32,
}

/// Shared top-k interface for the four indexes in this crate.
///
/// `score` is higher-is-better (we return approximate inner product). The
/// flat exact index returns true f32 inner product; the quantized indexes
/// return their unbiased estimator thereof.
pub trait Index {
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes of state the index holds beyond the raw originals.
    /// `FlatF32Index` reports `0` because its state *is* the originals.
    fn aux_bytes(&self) -> usize;
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;
}

/// Insert `cand` into a bounded max-heap of top-k results keyed by score
/// (higher = better). Uses `total_cmp` so NaN can never panic.
pub(crate) fn push_topk(heap: &mut Vec<SearchResult>, k: usize, cand: SearchResult) {
    if heap.len() < k {
        heap.push(cand);
        if heap.len() == k {
            heap.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
    } else if cand.score.total_cmp(&heap[k - 1].score).is_gt() {
        heap[k - 1] = cand;
        // Walk back up to its sorted position.
        let mut i = k - 1;
        while i > 0 && heap[i].score.total_cmp(&heap[i - 1].score).is_gt() {
            heap.swap(i, i - 1);
            i -= 1;
        }
    }
}

pub(crate) fn finalize_topk(mut heap: Vec<SearchResult>) -> Vec<SearchResult> {
    heap.sort_by(|a, b| b.score.total_cmp(&a.score));
    heap
}
