use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("dimension {dim} is not divisible by subspace count {m}")]
    DimNotDivisible { dim: usize, m: usize },

    #[error("expected vector of length {expected}, got {got}")]
    BadDim { expected: usize, got: usize },

    #[error("expected code of length {expected}, got {got}")]
    BadCode { expected: usize, got: usize },

    #[error("k={k} must satisfy 1 <= k <= 256 (codes are u8)")]
    BadK { k: usize },

    #[error("training set has {n} vectors but k={k}")]
    NotEnoughTrain { n: usize, k: usize },
}
