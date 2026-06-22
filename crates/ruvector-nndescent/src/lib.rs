//! # ruvector-nndescent
//!
//! Bulk kNN graph construction for HNSW base-layer bootstrap.
//!
//! Implements the **NN-Descent** algorithm (Dong, Charikar & Li, WWW 2011)
//! and a reverse-list extension (EFANNA 2016 style), then exposes a
//! trait-based `KnnGraphBuilder` so different bulk-construction backends
//! can be benchmarked head-to-head against the brute-force ground truth.
//!
//! The built kNN graph is used to seed the **layer-0** adjacency of a
//! minimal HNSW index — replacing N sequential `insert_layer0` calls
//! with one bulk graph build that's sub-quadratic in N.
//!
//! ## What's measured
//!
//! For each builder variant, the `benchmark` binary records:
//! - distance-computation count (algorithm-internal)
//! - wall-clock build time
//! - graph recall@K versus brute-force ground truth
//! - downstream search QPS / recall when the kNN graph is used as
//!   HNSW layer-0 adjacency
//!
//! All numbers in the research document are produced by a real
//! `cargo run --release --bin benchmark` invocation — no mocks.

pub mod dataset;
pub mod distance;
pub mod knn_graph;
pub mod brute;
pub mod nndescent;
pub mod hnsw_seeded;

pub use dataset::Dataset;
pub use distance::{l2_sq, DistanceCounter};
pub use knn_graph::{KnnGraph, KnnGraphBuilder, KnnNeighbor};
pub use brute::BruteForceBuilder;
pub use nndescent::{NnDescentBuilder, NnDescentConfig};
pub use hnsw_seeded::{SeededHnsw, SeededHnswConfig};
