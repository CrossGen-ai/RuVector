//! Full-precision (f32) L2 rerank — the ground truth baseline.

use crate::{sq_l2, Reranker};

pub struct L2Reranker {
    pub vectors: Vec<Vec<f32>>,
    pub dim: usize,
}

impl L2Reranker {
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        Self { vectors, dim }
    }
}

impl Reranker for L2Reranker {
    fn name(&self) -> &'static str {
        "FullL2"
    }
    fn bytes_per_vector(&self) -> usize {
        self.dim * std::mem::size_of::<f32>()
    }
    fn rerank(&self, query: &[f32], candidates: &[u32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = candidates
            .iter()
            .map(|&id| (sq_l2(query, &self.vectors[id as usize]), id))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }
}
