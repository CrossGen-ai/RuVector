use thiserror::Error;

#[derive(Debug, Error)]
pub enum EnsembleError {
    #[error("empty dataset")]
    EmptyDataset,
    #[error("rows have inconsistent dim: expected {expected}, got {actual}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("K must be in 1..=64, got {0}")]
    BadK(usize),
    #[error("query dim {query} != index dim {index}")]
    QueryDimMismatch { query: usize, index: usize },
}
