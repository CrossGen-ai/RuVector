//! LSH-seeded entry-point selection for small-world ANN graphs.
//!
//! A navigable-small-world (NSW) index normally starts every search from a
//! single fixed enterpoint and walks the graph greedily. On hard queries the
//! entry can be arbitrarily far from the true nearest cluster, so the beam
//! burns hops crossing the graph before it localizes.
//!
//! This crate pre-computes a short signed-random-hyperplane signature per
//! node and stores one representative node per LSH bucket. At search time
//! the query gets its own signature in O(sig_bits * dim) FMA, which is cheap
//! next to a greedy walk. The bucket's representative is used as the entry
//! seed. If the bucket is empty, we fall back to the fixed enterpoint, so
//! recall never drops below the baseline.
//!
//! Three entry strategies are exposed through [`EntryStrategy`] so the
//! benchmark can hold the graph, ef, and query stream constant and only vary
//! the seed-selection policy.

pub mod graph;
pub mod lsh;
pub mod metrics;
pub mod search;

pub use graph::{Nsw, NswParams};
pub use lsh::LshSeedIndex;
pub use metrics::{SearchStats, l2_sq};
pub use search::{EntryStrategy, SearchParams, search};
