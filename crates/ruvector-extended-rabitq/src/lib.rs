//! Extended RaBitQ (eRaBitQ): B-bit rotation-based vector quantization.
//!
//! Generalizes RaBitQ (1-bit) to B bits per coordinate using an
//! asymmetric distance estimator (query stays float, codes are B-bit).
//! Provides tight error bounds whose variance scales ~ 1 / (D * 2^(2B)).
//!
//! Pipeline (encode): x -> P x -> u = Px / ||Px|| -> code_i = Q_B(u_i)
//! Pipeline (query):  q -> P q  (no quantization)
//! Estimator: <q, x> ≈ ||x|| * <Pq, decode(code)>  where decode is
//! the codebook table applied per coordinate.
//!
//! The codebook is the optimal scalar quantizer for N(0, 1/D)
//! (per-coordinate marginal of a unit vector in D dim), truncated
//! to 2^B levels via Lloyd-Max iteration. Query-side look-ups cost
//! one load per coordinate, so decode is O(D).

pub mod codebook;
pub mod error_bound;
pub mod quantize;
pub mod rotation;
pub mod scan;

pub use codebook::Codebook;
pub use error_bound::VarianceBound;
pub use quantize::{EncodeStats, ExtendedRaBitQ, Index};
pub use rotation::{FastRotation, Rotation};
pub use scan::score;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("bit width {0} out of supported range 1..=8")]
    UnsupportedBits(u8),
    #[error("codebook needs at least 2 training vectors, got {0}")]
    InsufficientTraining(usize),
}

pub type Result<T> = std::result::Result<T, Error>;
