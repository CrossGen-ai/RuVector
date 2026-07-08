//! Error type for extended-rabitq.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExtRabitqError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("empty corpus: index needs at least one vector")]
    EmptyCorpus,

    #[error("unsupported bit width {bits}: only 1, 2, 4, 8 are supported")]
    UnsupportedBits { bits: u32 },

    #[error("invalid dimension {dim}: must be positive")]
    InvalidDim { dim: usize },

    #[error("index out of range: {index} >= {len}")]
    OutOfRange { index: usize, len: usize },
}
