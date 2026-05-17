use thiserror::Error;

#[derive(Debug, Error)]
pub enum SymphonyError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("dimension {d} not divisible by M={m}")]
    NotDivisible { d: usize, m: usize },
    #[error("empty input")]
    Empty,
    #[error("invalid parameter: {0}")]
    InvalidParam(&'static str),
}
