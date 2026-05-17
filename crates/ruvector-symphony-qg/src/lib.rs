//! ruvector-symphony-qg — Graph + Quantization fusion for ANN search.
//!
//! Three index backends share the [`AnnIndex`] trait so callers can swap
//! implementations and measure recall/throughput on the same harness:
//!
//! * [`FlatIndex`] — brute-force f32 (ground truth + baseline).
//! * [`PqRerankIndex`] — Product Quantization for candidate scoring, then
//!   full-precision reranking of the top-`r` (`r > k`).
//! * [`SymphonyQgIndex`] — small-world graph whose traversal scores edges
//!   with PQ distances, so the priority queue advances without a rerank
//!   pass. Optional [`SymphonyQgIndex::with_refine`] enables a tiny
//!   top-`refine` rescore for ablations.
//!
//! The PoC targets L2 distance in dimension `d`. Codewords are learned with
//! a deterministic k-means (seeded) and distances are computed via
//! asymmetric distance tables (ADT) — the standard PQ trick where the query
//! precomputes a `M × K` LUT once and per-vector distance is `M` table
//! lookups + adds.

pub mod error;
pub mod metric;
pub mod pq;
pub mod flat;
pub mod pq_rerank;
pub mod symphony;

pub use error::SymphonyError;
pub use flat::FlatIndex;
pub use pq::{ProductQuantizer, PqCodes};
pub use pq_rerank::PqRerankIndex;
pub use symphony::SymphonyQgIndex;

/// Common interface for ANN indices in this crate.
pub trait AnnIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}
