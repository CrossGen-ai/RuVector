//! # ruvector-anytime
//!
//! **Anytime HNSW** — progressive top-k refinement for interactive ANN search.
//!
//! Standard ANN search is a one-shot computation: the caller blocks until the
//! beam exhausts itself, then receives the final top-k. Many real workloads —
//! agent reasoning loops, interactive search UIs, latency-bounded RPC, and
//! streaming reranking pipelines — would prefer a **monotonically improving
//! stream of results** so they can act on the best guess available at any time
//! budget, then refine.
//!
//! This crate implements **anytime search** with a provable guarantee:
//!
//! > **Top-k Monotonicity.** For any two snapshots `S_i` and `S_j` emitted by an
//! > anytime searcher with `i ≤ j`, the multiset of `(node_id, dist)` in `S_j`
//! > is no worse than `S_i`: the farthest distance in `S_j` is `≤` the farthest
//! > in `S_i`, and `S_j`'s best-of-k is `≤` `S_i`'s best-of-k.
//!
//! Three variants quantify the throughput–reactivity trade-off:
//!
//! | Variant                | Snapshot policy                  | Overhead   |
//! |------------------------|----------------------------------|-----------:|
//! | [`OneShotSearch`]      | Final result only                | baseline   |
//! | [`NaiveAnytime`]       | Snapshot every pop               | high       |
//! | [`BatchedAnytime`]     | Snapshot on improvement, exp. backoff | low   |
//!
//! ## Why anytime?
//!
//! In an agent loop, the planner can dispatch a search with a soft deadline of
//! 8 ms but still benefit from any improved candidate emitted at 4 ms. In a
//! search UI, the first result can render instantly, then refine as the beam
//! deepens. In RPC, a server can return early when the SLA fires and the client
//! receives a calibrated best-effort answer instead of a deadline error.
//!
//! ## Relationship to HNSW
//!
//! As with `ruvector-coherence-hnsw`, the PoC operates on a single-layer flat
//! k-NN navigable small-world graph (HNSW layer-0 equivalent). The anytime
//! emission policy is orthogonal to layering and applies unchanged to a full
//! multi-layer HNSW.

pub mod dataset;
pub mod graph;
pub mod metrics;
pub mod search;

pub use graph::{FlatGraph, GraphConfig};
pub use search::{
    AnytimeSnapshot, BatchedAnytime, NaiveAnytime, OneShotSearch, SearchResult,
    Searcher,
};
