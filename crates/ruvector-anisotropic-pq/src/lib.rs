//! Anisotropic Product Quantization (ScaNN-style).
//!
//! Implements three swappable PQ backends behind a single `Quantizer` trait:
//!   * `Pq`              — plain product quantization (Lloyd k-means, symmetric L2 loss).
//!   * `Opq`             — Optimized PQ: learns an orthogonal rotation R so that
//!                         the rotated space aligns with subspace boundaries.
//!   * `AnisotropicPq`   — ScaNN's loss: weights parallel residual error higher
//!                         than orthogonal error during codebook training. For
//!                         maximum inner product / cosine search this lifts recall
//!                         vs plain PQ at the same code size.
//!
//! Provenance: the anisotropic loss is from Guo et al., "Accelerating Large-Scale
//! Inference with Anisotropic Vector Quantization", ICML 2020 (arXiv:1908.10396).
//! OPQ is Ge et al., "Optimized Product Quantization", CVPR 2013 / TPAMI 2014.
//! Plain PQ is Jégou, Douze, Schmid, "Product Quantization for NN Search", TPAMI 2011.

pub mod error;
pub mod kmeans;
pub mod metrics;
pub mod pq;
pub mod opq;
pub mod apq;
pub mod rotation;

pub use error::Error;
pub use pq::Pq;
pub use opq::Opq;
pub use apq::AnisotropicPq;

/// Common interface for the three product-quantizer variants.
pub trait Quantizer: Send + Sync {
    /// Encode a single vector to a packed code (one u8 per subspace).
    fn encode(&self, x: &[f32]) -> Vec<u8>;

    /// Asymmetric distance: caller passes the raw query, we score against
    /// a previously-encoded database vector. Distance is the metric this
    /// quantizer was trained for (squared-L2 for `Pq`/`Opq`, negative-inner-product
    /// for `AnisotropicPq`).
    fn asymmetric_score(&self, query: &[f32], code: &[u8]) -> f32;

    /// Memory cost per encoded vector, in bytes (not counting the codebook).
    fn code_bytes(&self) -> usize;

    /// Subspace count `m`, vector dim `d`, and centroids-per-subspace `k`.
    fn shape(&self) -> (usize, usize, usize);
}
