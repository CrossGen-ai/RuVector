//! Exact f32 top-k baseline (inner product). Serves as the recall oracle.

use crate::index::{finalize_topk, push_topk, Index, SearchResult};

#[derive(Clone)]
pub struct FlatF32Index {
    pub dim: usize,
    pub rows: Vec<Vec<f32>>,
}

impl FlatF32Index {
    pub fn build(rows: &[Vec<f32>]) -> Self {
        let dim = rows[0].len();
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.len(), dim, "row {i} has wrong dim");
        }
        Self {
            dim,
            rows: rows.to_vec(),
        }
    }
}

impl Index for FlatF32Index {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.rows.len()
    }
    fn aux_bytes(&self) -> usize {
        0
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim);
        let mut heap: Vec<SearchResult> = Vec::with_capacity(k);
        for (id, row) in self.rows.iter().enumerate() {
            let mut dot = 0.0_f32;
            for i in 0..self.dim {
                dot += row[i] * query[i];
            }
            push_topk(
                &mut heap,
                k,
                SearchResult {
                    id: id as u32,
                    score: dot,
                },
            );
        }
        finalize_topk(heap)
    }
}
