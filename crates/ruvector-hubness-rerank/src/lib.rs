//! ruvector-hubness-rerank
//!
//! Hubness-aware reranking for approximate nearest neighbor (ANN) search.
//!
//! ## Problem
//!
//! In high-dimensional vector spaces, the distance distribution concentrates:
//! most pairwise distances end up near the mean, and a small number of points
//! become **hubs** — they appear in the kNN list of a disproportionate number
//! of queries. Symmetrically, **anti-hubs** are almost never retrieved.
//! Radovanović et al. (2010) showed that hubness is an intrinsic property of
//! high-dim data and directly hurts kNN quality.
//!
//! Standard cosine / L2 reranking cannot fix this: the raw distance is
//! *exactly* what encodes the bias. To correct it, we rescale distances
//! using each point's own neighborhood statistics.
//!
//! ## Methods implemented
//!
//! - **Baseline** — raw Euclidean, no correction (control).
//! - **Local Scaling** (Zelnik-Manor & Perona 2004) — divides `d(q,v)` by the
//!   geometric mean of the k-th nearest-neighbor distance of `q` and of `v`.
//! - **Mutual Proximity — Empiric** (Schnitzer et al. 2012, JMLR 13:2871) —
//!   for each point `v`, precompute its distance distribution to all other
//!   points, then transform `d(q,v)` into the joint tail probability
//!   `MP(q,v) = P(d(q,·) > d(q,v)  ∧  d(v,·) > d(v,q))`. Uses an empirical
//!   CDF, so it is exact given the sample.
//! - **NICDM** (Non-Iterative Contextual Dissimilarity Measure, Jégou et al.
//!   2010) — divides `d(q,v)` by the *mean* of the top-k neighbor distances
//!   of `q` and of `v`. Cheaper than Mutual Proximity, similar effect.
//!
//! All methods share a common `Rerank` trait so an index can slot any of
//! them (or a custom one) behind a fixed API.
//!
//! ## Where this sits in the ANN pipeline
//!
//! 1. Coarse retrieval returns `R` candidates (`R = 4·k` typical) using the
//!    raw metric — cheap, biased.
//! 2. Reranker rescores those `R` candidates using the corrected score and
//!    returns the top `k`.
//!
//! The reranker touches only the shortlist, so it can be O(R log R) per query
//! and still improve end-to-end recall on the ground-truth kNN under the
//! *original* metric — because "raw top-k" and "true top-k" differ in
//! high-dim, and the corrected score is closer to the latter for a shared
//! notion of neighborhood consensus.

pub mod distance;
pub mod hubness;
pub mod index;
pub mod rerank;

pub use distance::{Distance, L2};
pub use hubness::{hubness_stats, HubnessStats};
pub use index::{FlatIndex, Vector};
pub use rerank::{
    BaselineRerank, LocalScalingRerank, MutualProximityRerank, NicdmRerank, Rerank, Scored,
};

/// Deterministic PRNG for reproducible benchmarks.
pub fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(seed)
}

#[cfg(test)]
mod smoke {
    use super::*;
    #[test]
    fn crate_reexports_visible() {
        let _l2 = L2;
        let _b = BaselineRerank;
    }
}
