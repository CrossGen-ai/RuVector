use thiserror::Error;

#[derive(Debug, Error)]
pub enum LvqError {
    #[error("empty training set")]
    EmptyTraining,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("unsupported bit width: {0} (must be 4 or 8)")]
    UnsupportedBits(u8),
}
