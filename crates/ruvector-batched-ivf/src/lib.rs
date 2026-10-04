//! Multi-query batched IVF scan with shared posting lists.
//!
//! Serving ANN queries one at a time wastes memory bandwidth: each query
//! fetches its own posting lists even when a sibling query in the batch
//! would touch the same lists. This crate groups a batch of queries by
//! probed centroid, scans every needed posting list exactly once, and
//! computes distances for all queries that need that list while the
//! vectors are hot in cache.
//!
//! The three strategies implemented share identical correctness
//! (same top-k set for the same probe lists) and differ only in how
//! the scan is scheduled:
//!   * `Serial`   — baseline, one query at a time, no sharing.
//!   * `BatchedListMajor` — group by posting list, scan each list once,
//!     update every subscribing query's heap.
//!   * `BatchedTiled` — list-major + a tiled inner loop so the posting
//!     list vectors stay in L1 while all subscribing queries are swept.

pub mod ivf;
pub mod scan;
pub mod batch;

pub use ivf::{IvfIndex, build_ivf};
pub use scan::{serial_scan, batched_list_major_scan, batched_tiled_scan};
pub use batch::{QueryBatch, TopK};
