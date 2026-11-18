//! NF4 (NormalFloat-4) Quantization for ANN Search.
//!
//! Inspired by QLoRA (Dettmers et al., 2023, arXiv:2305.14314), NF4 is an
//! information-theoretically optimal 4-bit data type for weights drawn from a
//! zero-mean normal distribution. Embedding vectors (OpenAI, Cohere, BGE,
//! E5, Matryoshka) are empirically close to zero-mean Gaussian after
//! centering, which makes NF4 a natural scalar-quantizer alternative to
//! linear SQ-4 and LVQ-4 for ANN search.
//!
//! This crate provides three trait-based variants:
//!
//! 1. [`FixedNf4`] — static codebook using 16 quantiles of N(0,1).
//! 2. [`AsymmetricNf4`] — per-vector scale (max-abs) and an asymmetric
//!    double-sided codebook with 16 levels (QLoRA's original NF4).
//! 3. [`LearnedNf4`] — Lloyd-Max-fitted codebook from a sample of the data
//!    set. Converges to the information-theoretic optimum for that data
//!    distribution (relaxes the "exactly Gaussian" assumption).
//!
//! All three implement [`Quantizer`]. Distances are computed with
//! asymmetric reconstruction: query stays FP32, database is 4-bit.

pub mod codebook;
pub mod quantizer;
pub mod variants;

pub use codebook::Codebook16;
pub use quantizer::{Quantizer, Code4};
pub use variants::{FixedNf4, AsymmetricNf4, LearnedNf4};
