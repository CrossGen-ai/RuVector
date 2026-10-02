//! Block-Max IVF-PQ scan.
//!
//! Ports the Block-Max WAND (BMW) idea from inverted-index retrieval to vector
//! search: posting lists are stored in fixed-size blocks, each block carries a
//! precomputed lower bound on the ADC distance between any query and any
//! vector in that block. During scan, blocks whose lower bound exceeds the
//! current top-K threshold are skipped wholesale.
//!
//! The crate is self-contained: an IVF-PQ index with three scan backends
//! sharing the same trait so recall and timing are directly comparable.

pub mod ivfpq;
pub mod scan;
pub mod topk;

pub use ivfpq::{IvfPq, IvfPqBuilder, PqParams};
pub use scan::{BlockMaxBitmaskScanner, BlockMaxScanner, BlockMaxTightScanner, Scanner, SequentialScanner, ScanStats};
pub use topk::{Hit, TopK};

/// Standard L2^2 between two slices.
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
