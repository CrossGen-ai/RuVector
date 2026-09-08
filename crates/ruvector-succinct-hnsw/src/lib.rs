//! ruvector-succinct-hnsw
//!
//! Memory-efficient NSW-style proximity-graph indices where the adjacency
//! store is a swappable [`Adjacency`] backend. Three concrete backends are
//! provided:
//!
//! - [`adjacency::DenseAdj`]         — baseline `Vec<Vec<u32>>`.
//! - [`adjacency::DeltaVarByteAdj`]  — sorted deltas + VarByte encoding
//!   packed into a single `Vec<u8>` blob with a `Vec<u32>` offset table.
//! - [`adjacency::ReorderedDeltaAdj`] — BFS re-permutes node ids so
//!   neighbours cluster in id space, *then* delta+VarByte encodes. This
//!   trades a one-shot O(N) permutation for markedly smaller deltas
//!   (~2× vs. plain delta on clustered corpora) and better cache
//!   behaviour during traversal.
//!
//! The core graph builder in [`graph`] is oblivious to the encoding and
//! calls back to the [`Adjacency`] trait for neighbour reads. Search
//! ([`search::beam_search`]) is likewise backend-agnostic.
//!
//! The design goal is to measure — not claim — memory / recall / latency
//! trade-offs across the three encodings on the same graph topology.
//! See [`bench`] for the harness driving both the integration test and
//! the `benchmark` binary.

pub mod adjacency;
pub mod bench;
pub mod graph;
pub mod reorder;
pub mod search;

/// A single vector in the corpus.
pub type Vector = Vec<f32>;

/// Node id in the graph (u32 to keep adjacency compact — 4 B per neighbour
/// before compression).
pub type NodeId = u32;

/// Distance functions the graph is generic over. We stick to squared L2
/// throughout: it is order-preserving with L2 and avoids the sqrt.
pub trait Distance: Send + Sync {
    fn dist(&self, a: &[f32], b: &[f32]) -> f32;
}

/// Squared Euclidean distance. Fine for kNN ranking (monotone in L2) and
/// slightly cheaper than L2.
#[derive(Clone, Copy, Default)]
pub struct SqEuclid;

impl Distance for SqEuclid {
    #[inline]
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut s = 0.0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }
}
