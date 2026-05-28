//! SPANN-style boundary-aware closure for IVF posting-list ANN.
//!
//! Three swappable assignment policies behind the `ClosurePolicy` trait:
//!   * `SingleAssign`     — baseline IVF (vector → 1 nearest centroid).
//!   * `FixedMultiAssign` — vector → k nearest centroids unconditionally.
//!   * `SpannClosure`     — SPANN (NeurIPS 2021): vector replicated into
//!     extra posting lists ONLY when its distance to the runner-up
//!     centroid is within `(1 + epsilon)` of its distance to the closest,
//!     bounded by `replica_cap`.
//!
//! References:
//!   - Chen et al., "SPANN: Highly-efficient Billion-scale Approximate
//!     Nearest Neighbor Search", NeurIPS 2021.
//!   - Jégou et al., "Product Quantization for Nearest Neighbor Search",
//!     IEEE TPAMI 2011 (baseline IVF formulation).

pub mod kmeans;
pub mod index;
pub mod policy;
pub mod metrics;

pub use index::{SpannIndex, SearchResult};
pub use policy::{ClosurePolicy, SingleAssign, FixedMultiAssign, SpannClosure, PolicyKind};
pub use metrics::sq_l2;
