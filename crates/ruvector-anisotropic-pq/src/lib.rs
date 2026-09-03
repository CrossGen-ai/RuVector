//! # ruvector-anisotropic-pq
//!
//! **Anisotropic score-aware product quantization (ScaNN loss).**
//!
//! Standard product quantization (PQ) trains codebooks by minimizing squared
//! reconstruction error `||x - x̃||²`. That objective treats every direction of
//! the residual as equally important. But when the *downstream* task is
//! maximum-inner-product search (MIPS) — the common case for embedding-based
//! retrieval — errors along the query direction hurt the ranking score
//! quadratically more than errors orthogonal to the query.
//!
//! Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector
//! Quantization* (ICML 2020, arXiv:1908.10396), formalise this with a
//! score-aware loss:
//!
//! ```text
//! L_η(x, x̃) = η · ||r_∥||²  +  ||r_⊥||²
//! ```
//!
//! where `r = x̃ - x` is the residual, `r_∥` is its component along the unit
//! datapoint direction `x̂ = x / ||x||`, and `r_⊥ = r - r_∥`. The weight
//! `η ≥ 1` upweights the parallel component. `η = 1` recovers ordinary PQ.
//! Real ScaNN uses larger `η` (often 4–16) and reports substantial MIPS
//! recall gains at fixed code size.
//!
//! ## What this crate implements
//!
//! A minimal, honest PoC that isolates the *score-aware loss idea* from the
//! rest of ScaNN's engineering (partitioning, SIMD LUTs, in-register lookup).
//! Three swappable quantizers behind one trait:
//!
//! | Quantizer | η | Centroid update |
//! |-----------|---|-----------------|
//! | [`BaselinePq`] | 1 | Unweighted mean (standard k-means) |
//! | [`AnisotropicPq`] with `eta = 4.0` | 4 | Weighted least squares |
//! | [`AnisotropicPq`] with `eta = 16.0` | 16 | Weighted least squares |
//!
//! Search uses asymmetric distance computation (ADC): each query builds an
//! `M × K` lookup table of query-to-centroid inner products; the code of each
//! database vector then indexes the table and sums the `M` partial scores.
//!
//! ## Measured PoC outcome
//!
//! On a deterministic 5 000-point / 500-query Gaussian-mixture dataset with
//! `dim = 64`, `M = 8`, `K = 256`, MIPS Recall@10:
//!
//! | Quantizer | Recall@10 | Reconstruction MSE |
//! |-----------|-----------|--------------------|
//! | BaselinePq (η=1) | see benchmark output | best |
//! | AnisotropicPq (η=4) | ≥ baseline | slightly worse |
//! | AnisotropicPq (η=16) | ≥ baseline | worst |
//!
//! Anisotropic training *should* beat baseline on MIPS recall while giving up
//! some Euclidean-reconstruction MSE — that is the whole point. See the
//! research README for exact numbers from the last run.
//!
//! ## RuVector fit
//!
//! Every ANN index in the workspace that stores compressed vectors for MIPS
//! (RaBitQ, PQ-search, DiskANN residual codes, MaxSim per-token codes) pays
//! the isotropic-PQ tax. This crate provides a drop-in codebook trainer whose
//! output has the same on-disk layout as ordinary PQ but different centroid
//! values — swappable at index build time with zero query-path change.

pub mod dataset;
pub mod metrics;
pub mod pq;
pub mod search;

pub use dataset::{DatasetConfig, GaussianMixture};
pub use metrics::{recall_at_k, reconstruction_mse};
pub use pq::{AnisotropicPq, BaselinePq, Codebook, PqParams, Quantizer};
pub use search::{AdcSearcher, Hit};
