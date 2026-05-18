//! Error types for `ruvector-roargraph`.

use std::fmt;

/// All errors that can be returned by `ruvector-roargraph`.
#[derive(Debug, Clone, PartialEq)]
pub enum RoarError {
    /// Attempt to call `search` or `build` before `add` has populated the index.
    EmptyIndex,
    /// Dimension of a query or base vector does not match the index dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// The training query set is empty; at least one query is required.
    NoTrainingQueries,
    /// Requested `k` is larger than the number of indexed vectors.
    KTooLarge { k: usize, n: usize },
    /// `build` has not been called yet; the graph has no edges.
    NotBuilt,
}

impl fmt::Display for RoarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoarError::EmptyIndex => write!(f, "index is empty — call add() before build/search"),
            RoarError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            RoarError::NoTrainingQueries => {
                write!(f, "training query set is empty")
            }
            RoarError::KTooLarge { k, n } => {
                write!(f, "k={k} exceeds number of indexed vectors n={n}")
            }
            RoarError::NotBuilt => {
                write!(f, "graph not built — call build() before search()")
            }
        }
    }
}

impl std::error::Error for RoarError {}
