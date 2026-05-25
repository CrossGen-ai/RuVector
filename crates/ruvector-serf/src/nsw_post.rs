//! Single global NSW + postfilter baseline.
//!
//! Build one graph over the whole dataset, search with an inflated `k`, then
//! drop anything outside the requested range. Recall degrades sharply when
//! the range is selective.

use crate::nsw::{Nsw, NswParams};
use crate::{Range, RangeAnn};
use std::sync::Arc;

pub struct NswPost {
    pub keys: Vec<f32>,
    pub graph: Nsw,
    pub overscan: usize,
}

impl NswPost {
    pub fn build(
        vectors: Arc<Vec<Vec<f32>>>,
        keys: Vec<f32>,
        params: NswParams,
        overscan: usize,
    ) -> Self {
        assert_eq!(vectors.len(), keys.len());
        let ids: Vec<u32> = (0..vectors.len() as u32).collect();
        let graph = Nsw::build(vectors, ids, params);
        Self {
            keys,
            graph,
            overscan,
        }
    }

    pub fn graph_bytes(&self) -> usize {
        self.graph.adj_bytes()
    }
}

impl RangeAnn for NswPost {
    fn name(&self) -> &'static str {
        "nsw-postfilter"
    }
    fn search(&self, q: &[f32], range: Range, k: usize) -> Vec<(usize, f32)> {
        let pulled = self.graph.search(q, k * self.overscan);
        let mut filt: Vec<(usize, f32)> = pulled
            .into_iter()
            .filter(|(i, _)| range.contains(self.keys[*i]))
            .collect();
        filt.truncate(k);
        filt
    }
}
