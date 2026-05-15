//! MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings.
//!
//! Reference: Dhulipala, Hadian, Jayaram, Lee, Mirrokni — "MUVERA:
//! Multi-Vector Retrieval via Fixed Dimensional Encodings"
//! (NeurIPS 2024, arXiv:2405.19504).
//!
//! Given a multi-vector representation (e.g. ColBERT/ColPali tokens),
//! MUVERA produces a *single* fixed-dim vector whose inner product is an
//! unbiased estimator of the asymmetric Chamfer similarity
//!
//! ```text
//! Chamfer(Q, D) = sum_{q in Q} max_{d in D} <q, d>
//! ```
//!
//! This lets you reuse any single-vector ANN index (HNSW, IVF, RaBitQ,
//! AnisotropicVQ — all of which already live in this workspace) for what
//! used to require expensive late-interaction scoring.
//!
//! # Algorithm
//! 1. Draw `k_sim` random Gaussian hyperplanes; SimHash assigns each token
//!    to one of `B = 2^k_sim` buckets.
//! 2. For a document, *sum* (or mean) all token vectors landing in each
//!    bucket → produces a `(B * d)` block. For a query, also sum token
//!    vectors per bucket; missing query buckets are filled by the nearest
//!    non-empty document bucket via the SimHash partition (the original
//!    paper calls this "fill"). For symmetric Chamfer estimation we use
//!    the *fill rule* on the side that is shorter.
//! 3. Repeat with `R_reps` independent SimHash projections, concatenate.
//!    Final dim is `R * B * d`.
//! 4. Optional random projection (count-sketch / Gaussian) compresses the
//!    concat to `d_final` bytes — the paper shows tiny accuracy loss.
//!
//! # Why it matters
//! ColBERT-v2 / ColPali style late interaction is expensive (O(|Q|*|D|*d))
//! per pair. MUVERA reduces it to *one* dot product per pair after a
//! per-document one-time encoding, with provable distortion bounds.

#![forbid(unsafe_code)]

pub mod encoder;
pub mod error;
pub mod metrics;

pub use encoder::{FdeConfig, FdeEncoder, FillStrategy, ProjectionMode};
pub use error::MuveraError;
pub use metrics::chamfer_similarity;
