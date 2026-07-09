//! # ruvector-aisaq
//!
//! AISAQ — All-in-Storage ANNS with Quantization.
//!
//! This crate is a Rust proof-of-concept of the AISAQ design
//! (Kioxia, arXiv:2404.06004, 2024), adapted for the ruvector
//! ecosystem. AISAQ is a variant of DiskANN-style graph search
//! where **both** the raw vectors *and* the product-quantization
//! (PQ) codes live on SSD. Only the navigation graph stays in
//! RAM, so the memory footprint is dominated by `N * degree * 4`
//! bytes instead of `N * (D * 4 + PQ_M)` bytes.
//!
//! The crate exposes a small [`DistanceBackend`] trait so we can
//! swap between three real, measurable variants:
//!
//! * [`backends::FlatF32Ram`]  — raw f32 vectors held in RAM (baseline)
//! * [`backends::PqRam`]       — PQ codes held in RAM (in-memory ADC)
//! * [`backends::PqDisk`]      — PQ codes memory-mapped from disk (AISAQ)
//!
//! All three back the same [`graph::BeamSearcher`], which lets the
//! benchmark isolate the storage decision from the search algorithm.
//!
//! The PoC is intentionally single-threaded and free of unsafe
//! outside of the memmap read path — the goal is *correct* numbers,
//! not a production kernel. A production layout is discussed in
//! `docs/research/nightly/2026-07-09-aisaq-all-in-storage-quantization/README.md`.

pub mod pq;
pub mod graph;
pub mod backends;

pub use backends::{DistanceBackend, FlatF32Ram, PqRam, PqDisk};
pub use graph::{KnnGraph, BeamSearcher};
pub use pq::ProductQuantizer;

/// Compute L2^2 distance between two equal-length f32 slices.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Compute recall@k of `predicted` against `truth` (both length k).
pub fn recall_at_k(predicted: &[u32], truth: &[u32]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let mut hits = 0usize;
    for t in truth {
        if predicted.contains(t) {
            hits += 1;
        }
    }
    hits as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_sq_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 4.0];
        assert!((l2_sq(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_perfect() {
        assert_eq!(recall_at_k(&[1, 2, 3], &[3, 2, 1]), 1.0);
    }

    #[test]
    fn recall_half() {
        assert_eq!(recall_at_k(&[1, 2], &[1, 9]), 0.5);
    }
}
