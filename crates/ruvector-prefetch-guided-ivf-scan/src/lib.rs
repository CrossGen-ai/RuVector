//! # ruvector-prefetch-guided-ivf-scan
//!
//! Software-prefetch-guided IVF (Inverted File) posting-list scan for ANN,
//! designed as a drop-in kernel for RuVector's IVF-flat and IVF-PQ paths.
//! Portable across x86_64 (`_mm_prefetch`) and aarch64 (`PRFM` via stable
//! inline assembly). On other targets the prefetch calls compile down to
//! no-ops.
//!
//! ## Why this crate exists
//!
//! Modern IVF scans are memory-bound, not compute-bound: at dim=128 the L2²
//! kernel takes ~20 cycles of ALU work per vector, but the vector itself is
//! 512 bytes = 8 cache lines. On a system where L2 miss cost dwarfs compute
//! (Apple M-series, Xeon, Graviton), the CPU stalls waiting for the *next*
//! vector while ALUs sit idle. The hardware stream prefetcher helps for
//! contiguous scans of small-dim vectors but starts to fall behind once
//! `dim * 4` bytes exceeds ~2 cache lines, and it can't help at all across
//! non-contiguous cluster boundaries.
//!
//! Software prefetch closes the gap: issue `PRFM` / `_mm_prefetch` for
//! vector `i + K` while computing distance for vector `i`. Choose `K` from
//! the vector byte size, and the same code auto-tunes across `dim`.
//!
//! ## API
//!
//! ```no_run
//! use ruvector_prefetch_guided_ivf_scan::{PostingList, scan_adaptive_prefetch};
//!
//! let dim = 128;
//! let n = 10_000;
//! let vectors = vec![0.0f32; n * dim];
//! let list = PostingList::new(dim, vectors);
//! let query = vec![0.0f32; dim];
//! let top = scan_adaptive_prefetch(&list, &query, 10);
//! assert!(top.len() <= 10);
//! ```

pub mod distance;
pub mod prefetch;
pub mod scan;

pub use distance::l2_sq;
pub use prefetch::{adaptive_lookahead, prefetch_read, prefetch_slice_l1, Locality};
pub use scan::{
    scan_adaptive_prefetch, scan_fixed_prefetch, scan_no_prefetch, scan_strided_no_prefetch,
    scan_strided_prefetch, Hit, PostingList, TopK,
};
