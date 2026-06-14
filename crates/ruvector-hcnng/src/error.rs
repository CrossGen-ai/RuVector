use thiserror::Error;

#[derive(Debug, Error)]
pub enum HcnngError {
    #[error("empty dataset")]
    Empty,
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
    #[error("invalid parameter: {0}")]
    InvalidParam(&'static str),
    #[error("serde: {0}")]
    Serde(String),
}
