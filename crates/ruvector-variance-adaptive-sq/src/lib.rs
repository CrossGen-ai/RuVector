//! Variance-Adaptive Scalar Quantization (VA-SQ).
//!
//! Scalar quantization normally spends the same number of bits on every
//! dimension of the vector. Real embedding distributions are not uniform:
//! a handful of dimensions carry most of the variance, and spending the
//! same budget on near-constant dimensions wastes bits.
//!
//! VA-SQ allocates a *global* bit budget `B` across `D` dimensions by a
//! variance-proportional rule, clipped to [b_min, b_max]:
//!
//! ```text
//! b_i  ∝  log2(sigma_i^2 + eps),  sum b_i = B * D
//! ```
//!
//! At encode time each dimension is independently min-max quantized to its
//! own level count 2^{b_i}. Codes are bit-packed into a flat byte buffer.
//! Reconstruction is a per-dimension LUT lookup. All distances in this
//! crate are L2 (squared) between a float query and reconstructed codes.
//!
//! See the research doc at `docs/research/nightly/2026-10-28-variance-adaptive-sq`.

pub mod alloc;
pub mod codec;
pub mod distance;
pub mod dataset;

pub use alloc::{allocate_bits, AllocPolicy};
pub use codec::{VaSqCodec, VaSqConfig};
pub use dataset::{gen_lognormal_embeddings, Dataset};
pub use distance::{brute_force_topk, l2_sq, recall_at_k};
