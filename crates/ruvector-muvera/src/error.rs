use thiserror::Error;

#[derive(Debug, Error)]
pub enum MuveraError {
    #[error("dimension mismatch: encoder expects d={expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },

    #[error("empty multi-vector: at least one token vector required")]
    EmptyMultiVector,

    #[error("invalid config: {0}")]
    InvalidConfig(&'static str),
}
