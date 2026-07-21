//! # ruvector-elastic-pq
//!
//! Elastic Bit Allocation Product Quantization (EBA-PQ).
//!
//! Standard Product Quantization (Jégou et al. 2011) partitions each vector
//! into `M` subvectors and quantizes each subvector with a codebook of
//! `K = 2^b` centroids, where `b` (bits per subspace) is uniform. On real
//! embedding data the per-subspace variance is highly non-uniform; giving
//! every subspace the same bit budget wastes bits on quiet subspaces and
//! starves the noisy ones — inflating quantization distortion.
//!
//! `EBA-PQ` keeps the total bit budget `B = M * b` fixed but distributes
//! bits across subspaces proportionally to their post-training codebook
//! distortion. We support three allocation strategies:
//!
//! 1. [`Allocator::Uniform`] — the classic PQ baseline (`b` bits every
//!    subspace).
//! 2. [`Allocator::VarianceProportional`] — bits scale with each
//!    subspace's variance (PCA-style prior).
//! 3. [`Allocator::DistortionIterative`] — start uniform, retrain, then
//!    move one bit at a time from the lowest-distortion-per-bit subspace
//!    to the highest until the exchange stops paying off. This is the
//!    "elastic" variant; it is what the ADR calls out as the novel
//!    contribution.
//!
//! ## Non-Goals
//!
//! * We do not learn a rotation (that would make this OPQ). Rotation
//!   composes cleanly with EBA-PQ but is orthogonal to bit allocation.
//! * We do not implement SIMD ADC scan; the goal is correctness plus
//!   representative benchmark numbers, not the absolute floor.
//!
//! ## Traits
//!
//! [`Quantizer`] is the swappable trait every allocator plugs into so
//! future variants (learned bit allocators, entropy-coded codebooks) can
//! be dropped in without changing search-side code.

#![deny(rust_2018_idioms)]
#![warn(missing_docs)]

mod allocator;
mod codebook;
mod kmeans;
mod pq;
mod search;

pub use allocator::{Allocator, BitBudget};
pub use codebook::Codebook;
pub use pq::{ElasticPq, ElasticPqBuilder, ElasticPqError, TrainStats};
pub use search::{AdcSearcher, SearchResult};

/// Swappable interface for any PQ-style quantizer that this crate can
/// benchmark against EBA-PQ.
pub trait Quantizer: Send + Sync {
    /// Encode a single vector into the packed byte code produced by the
    /// implementation.
    fn encode(&self, vec: &[f32]) -> Vec<u8>;

    /// Approximate the squared L2 distance between `query` and a code
    /// previously produced by [`Quantizer::encode`].
    fn asym_l2_sq(&self, query: &[f32], code: &[u8]) -> f32;

    /// Total number of bits used per encoded vector. This is what the
    /// bit budget comparison uses to keep methods on the same footing.
    fn code_bits(&self) -> usize;

    /// Human name for the benchmark table.
    fn name(&self) -> &'static str;
}
