//! Graph + vector storage in CSR form.

pub type Vector = Vec<f32>;

/// A single-layer HNSW-style k-NN proximity graph in CSR form.
/// (Multi-layer HNSW reduces to a base-layer graph for the purpose of
/// reordering; the base layer is where the vast majority of search
/// cache misses occur.)
#[derive(Clone, Debug)]
pub struct HnswGraph {
    pub dim: usize,
    /// Vector store, row-major, length = n * dim.
    pub data: Vec<f32>,
    /// CSR neighbour offsets, length n+1.
    pub offsets: Vec<u32>,
    /// CSR neighbour indices.
    pub neighbours: Vec<u32>,
    /// Entry point for greedy search.
    pub entry: u32,
}

impl HnswGraph {
    pub fn n(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub fn neighbours_of(&self, i: u32) -> &[u32] {
        let s = self.offsets[i as usize] as usize;
        let e = self.offsets[i as usize + 1] as usize;
        &self.neighbours[s..e]
    }

    #[inline]
    pub fn vector(&self, i: u32) -> &[f32] {
        let s = i as usize * self.dim;
        &self.data[s..s + self.dim]
    }

    /// Total bytes touched by a full graph traversal (sizeof adjacency + vectors).
    pub fn bytes(&self) -> usize {
        self.data.len() * 4 + self.neighbours.len() * 4 + self.offsets.len() * 4
    }
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
