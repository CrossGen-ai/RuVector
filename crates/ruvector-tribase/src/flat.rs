//! Baseline: exhaustive flat search. Returns top-k by squared L2.

use crate::dist::{sq_l2, TopK};

pub struct FlatIndex {
    pub data: Vec<Vec<f32>>,
    pub dim: usize,
}

#[derive(Default, Clone, Copy)]
pub struct SearchStats {
    pub dist_computations: u64,
    pub vectors_scanned: u64,
    pub lists_scanned: u64,
}

impl FlatIndex {
    pub fn build(data: Vec<Vec<f32>>) -> Self {
        let dim = data[0].len();
        Self { data, dim }
    }

    pub fn search(&self, q: &[f32], k: usize) -> (Vec<(f32, u32)>, SearchStats) {
        let mut heap = TopK::new(k);
        for (i, x) in self.data.iter().enumerate() {
            heap.push(sq_l2(q, x), i as u32);
        }
        let stats = SearchStats {
            dist_computations: self.data.len() as u64,
            vectors_scanned: self.data.len() as u64,
            lists_scanned: 1,
        };
        (heap.into_sorted(), stats)
    }
}
