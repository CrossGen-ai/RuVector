//! SymphonyQG — joint graph + 1-bit quantization ANN index.
//!
//! Inspired by Yang et al., "SymphonyQG: Towards Symphonious Integration of
//! Quantization and Graph for Approximate Nearest Neighbor Search" (SIGMOD 2025).
//!
//! Three backends share the same NSW-style graph, exposed via [`Searcher`]:
//!   * `Float`     — float L2 baseline (oracle on the graph).
//!   * `Binary`    — 1-bit RaBitQ-style code; popcount-driven traversal.
//!   * `Symphony`  — quantized traversal + float rerank of top-`r` (the actual
//!     SymphonyQG recipe; cheap candidates, exact final ordering).

#![forbid(unsafe_code)]

pub mod graph;
pub mod quant;
pub mod symphony;

pub use graph::{Graph, GraphBuilder};
pub use quant::{BinaryCodec, BinaryCode};
pub use symphony::{Searcher, SearchMode, Neighbor};

/// L2² between two equal-length float slices. Public for tests / benches.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
