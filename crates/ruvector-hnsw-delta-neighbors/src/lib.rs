//! ruvector-hnsw-delta-neighbors
//!
//! Compressed neighbor-list storage for HNSW-style graphs.
//!
//! The **graph** structure (which node connects to which) dominates HNSW
//! memory once vectors are quantized (PQ, RaBitQ, INT8). A flat `Vec<u32>`
//! neighbor list uses 4 bytes per edge; at M=32 that is 128 bytes/node just
//! for the base layer, before any vector data. This crate implements two
//! encoded variants that operate on *sorted* neighbor lists (HNSW does not
//! require neighbor order to be preserved) after an optional
//! locality-preserving remap of node IDs:
//!
//! * [`FlatU32Store`]   — baseline, 4 bytes/edge.
//! * [`VarintDeltaStore`] — sorted-delta + LEB128 varints.
//! * [`BitpackedDeltaStore`] — sorted-delta + per-list bit-packed fixed width.
//!
//! A [`NeighborStore`] trait abstracts them so downstream HNSW code can swap
//! backends. Every impl decodes into a caller-owned `SmallVec`-style buffer
//! (here a `&mut Vec<u32>`) so hot search paths avoid allocation.
//!
//! ID remapping: [`locality_remap_bfs`] performs a BFS from an arbitrary seed
//! and produces a permutation that places graph neighbors close in the new ID
//! space. On typical HNSW graphs this shrinks median delta magnitudes by an
//! order of magnitude, which is what makes varint/bit-packing pay off.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod graph;
pub mod remap;
pub mod stores;

pub use graph::{brute_knn_graph, Graph};
pub use remap::{apply_permutation, locality_remap_bfs};
pub use stores::{BitpackedDeltaStore, FlatU32Store, NeighborStore, VarintDeltaStore};

/// Convenience: build all three stores from the same graph and report bytes.
#[derive(Debug, Clone, Copy)]
pub struct StoreSizes {
    /// Bytes used by [`FlatU32Store`] (baseline).
    pub flat: usize,
    /// Bytes used by [`VarintDeltaStore`].
    pub varint: usize,
    /// Bytes used by [`BitpackedDeltaStore`].
    pub bitpacked: usize,
}

impl StoreSizes {
    /// Ratio of `flat / varint`.
    pub fn varint_ratio(&self) -> f64 {
        self.flat as f64 / self.varint.max(1) as f64
    }
    /// Ratio of `flat / bitpacked`.
    pub fn bitpacked_ratio(&self) -> f64 {
        self.flat as f64 / self.bitpacked.max(1) as f64
    }
}

/// Compute sizes for all three encodings against `graph`.
pub fn measure_all(graph: &Graph) -> StoreSizes {
    let flat = FlatU32Store::from_graph(graph).bytes();
    let varint = VarintDeltaStore::from_graph(graph).bytes();
    let bitpacked = BitpackedDeltaStore::from_graph(graph).bytes();
    StoreSizes { flat, varint, bitpacked }
}
