//! Bounded Early-Termination IVF (BET-IVF).
//!
//! Per-query adaptive nprobe for IVF using triangle-inequality lower bounds on
//! unvisited partitions. Stops scanning when the worst current top-k distance
//! beats the tightest achievable distance from any remaining partition.

use thiserror::Error;

pub mod ivf;
pub mod search;
#[cfg(test)]
mod tests;

pub use ivf::{IvfIndex, Partition};
pub use search::{SearchStats, SearchStrategy, search};

#[derive(Debug, Error)]
pub enum BetIvfError {
    #[error("dimension mismatch: index dim {0}, query dim {1}")]
    DimensionMismatch(usize, usize),
    #[error("empty dataset")]
    EmptyDataset,
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, BetIvfError>;

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    l2_sq(a, b).sqrt()
}
