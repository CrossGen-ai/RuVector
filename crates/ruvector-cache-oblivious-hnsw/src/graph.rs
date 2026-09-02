//! Flat HNSW graph representation, permutable by node-id → slot mapping.

use crate::DIM;

/// Which physical layout the [`FlatGraph`] uses. Recorded for reporting only;
/// search behaviour is identical across variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Bfs,
    Dfs,
    Veb,
}

/// HNSW graph flattened into contiguous arrays. Every field is indexed by
/// the *permuted slot*, not the logical node id.
///
/// `perm[logical_id] = slot`, so callers translate ids exactly once at
/// query entry; the entire hot path deals in slots.
pub struct FlatGraph {
    pub layout: Layout,
    /// `dim`-major flat vector storage, `n * DIM` floats.
    pub vectors: Vec<f32>,
    /// Neighbour ids per node, `M` slots per node in slot order.
    /// Empty slots are marked with `u32::MAX`.
    pub neighbours: Vec<u32>,
    /// Max degree used by the layout (both `M` and `M0` folded — this is a
    /// research skeleton with one level).
    pub m: usize,
    /// Permutation: `perm[logical_id] = slot`.
    pub perm: Vec<u32>,
    /// Inverse permutation: `inv[slot] = logical_id`.
    pub inv: Vec<u32>,
    /// Slot of the entry point (permuted).
    pub entry: u32,
}

impl FlatGraph {
    pub fn n(&self) -> usize {
        self.vectors.len() / DIM
    }

    #[inline(always)]
    pub fn vector(&self, slot: u32) -> &[f32] {
        let start = (slot as usize) * DIM;
        &self.vectors[start..start + DIM]
    }

    #[inline(always)]
    pub fn neighbours_of(&self, slot: u32) -> &[u32] {
        let start = (slot as usize) * self.m;
        &self.neighbours[start..start + self.m]
    }
}
