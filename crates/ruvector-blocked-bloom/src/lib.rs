//! # ruvector-blocked-bloom
//!
//! Fast, cache-line-aligned **visited-set** structures for HNSW / Vamana / DEG
//! graph traversal.  The purpose of a visited-set is to answer *"have I seen
//! node `id` on this query?"* — a hot path called once per neighbor visit,
//! which typically dominates search allocation and TLB traffic.
//!
//! This crate provides three interchangeable implementations behind a single
//! [`VisitedSet`] trait:
//!
//! | Impl                     | Exact | Bytes / query          | Notes                                                                     |
//! |--------------------------|-------|------------------------|---------------------------------------------------------------------------|
//! | [`HashSetVisited`]       | yes   | ~48 · visited          | Baseline `HashSet<u32>`                                                   |
//! | [`BitmapVisited`]        | yes   | `n_ids / 8` (fixed)    | Dense bitmap indexed by node id — fastest exact                            |
//! | [`BlockedBloomVisited`]  | no    | 64 · n_blocks (fixed)  | Cache-line-aligned blocked Bloom filter, 4 hashes per 512-bit block        |
//!
//! `BlockedBloomVisited` sacrifices exactness for a tunable false-positive
//! rate.  Because HNSW search is already approximate, and a false-positive
//! visited only causes an **early prune** (not a wrong result), the recall
//! impact is small and measurable.  The upshot is a fixed-size, allocation-
//! free structure with two cache lines touched per probe — well below the
//! ~5–20 cache lines of a `HashSet<u32>` probe including rehashing.
//!
//! See `examples/hnsw_walk.rs` for a query-frontier simulation and
//! `src/benchmark.rs` for the multi-variant benchmark.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod bitmap;
pub mod bloom;
pub mod hashset;

pub use bitmap::BitmapVisited;
pub use bloom::{BlockedBloomVisited, BloomStats};
pub use hashset::HashSetVisited;

/// Universal contract for HNSW / graph-traversal visited-sets.
///
/// Implementations must be cheap to `reset` between queries and safe to
/// call from a single thread.  Returning `true` from [`mark`] means
/// *"newly inserted"* — i.e. the caller should proceed to process the id
/// as a fresh neighbor.
pub trait VisitedSet {
    /// Attempt to mark `id` visited.  Returns `true` iff the id was
    /// (probabilistically) not previously seen and should be processed.
    fn mark(&mut self, id: u32) -> bool;

    /// Query without inserting.  Returns `true` iff `id` is (probabilistically)
    /// already visited.
    fn contains(&self, id: u32) -> bool;

    /// Reset for the next query.  Must run in O(structure) not O(inserts_seen).
    fn reset(&mut self);

    /// Rough memory usage in bytes — used for reporting only.
    fn bytes(&self) -> usize;

    /// Human-readable name for benchmark tables.
    fn name(&self) -> &'static str;
}
