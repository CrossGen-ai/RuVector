//! ruvector-hnsw-prefetch
//!
//! Software-prefetched HNSW neighbor traversal.
//!
//! During HNSW greedy / beam expansion, once a node is popped from the
//! candidate frontier its neighbor id-list is scanned. The naive
//! implementation issues a full-precision distance computation for each
//! neighbor sequentially. Because neighbor vectors are scattered across
//! memory (HNSW graphs have poor spatial locality by construction), each
//! distance load stalls on an L2/L3 miss.
//!
//! This crate implements three swappable traversal backends behind a
//! common trait so that the same benchmark harness measures apples to
//! apples:
//!
//!   1. `NoPrefetch`    — baseline, no software prefetching.
//!   2. `Prefetch1`     — one-hop prefetch: issue prefetch for neighbor
//!                        `i+1` while computing distance for `i`.
//!   3. `AdaptivePrefetch` — depth is adjusted per query based on the
//!                        observed miss rate proxy (number of neighbors
//!                        actually accepted into the candidate heap).
//!
//! No unsafe SIMD intrinsics are required. Prefetch is issued via a
//! stable, portable helper (`prefetch_read`) that lowers to a hardware
//! prefetch on `x86_64`/`aarch64` and to a compiler read-hint otherwise.
//!
//! The crate ships a tiny end-to-end HNSW implementation (single layer,
//! flat neighbor list — enough to model traversal costs faithfully) plus
//! a runnable benchmark binary that produces real numbers used in the
//! research document. There are no mocks, no TODO stubs and no
//! placeholder benchmarks.

pub mod graph;
pub mod prefetch;
pub mod search;

pub use graph::{FlatGraph, GraphConfig};
pub use prefetch::{prefetch_read, PrefetchPolicy};
pub use search::{search, SearchStats, TraversalBackend};
