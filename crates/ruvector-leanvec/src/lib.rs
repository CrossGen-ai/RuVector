//! ruvector-leanvec — LeanVec-LVQ
//!
//! Two compositional ideas from Intel Labs' SVS line of work:
//!
//! 1. **LVQ — Locally-adaptive Vector Quantization.** Each vector gets its own
//!    affine scale `(lo, step)` so 8-bit codes recover values in
//!    `[lo, lo + 255·step]`. Asymmetric distance: query stays f32, database is
//!    decoded on the fly. See Aguerrebere et al., 2023 ("Similarity search in
//!    the blink of an eye with compressed indices").
//! 2. **LeanVec — learned linear projection.** A trained orthonormal projection
//!    `P ∈ R^{r×d}` (here PCA on a sample) shrinks each vector from `d` to `r`
//!    dims before LVQ. Original vectors are kept on the side for an exact L2
//!    rerank of the top-k' candidates. See Tepper et al., 2024 ("LeanVec:
//!    Searching vectors faster by making them fit").
//!
//! The three knobs — projection rank `r`, LVQ bits `b`, and rerank fan-out
//! `k'` — let you trade memory and speed against recall. This crate ships
//! three runnable variants exercised by the demo binary and integration tests:
//!
//! | Variant      | per-vector bytes (d=128) | recall@10 | wall ns / query |
//! |--------------|--------------------------|-----------|-----------------|
//! | Flat f32     | 512                      | 1.000     | reference        |
//! | LVQ-8        | 128 + 8                  | high      | ~4× over flat   |
//! | LeanVec(r/2) | 64 + 8 + raw rerank      | high      | ~6× over flat   |
//!
//! Real numbers in `docs/research/nightly/2026-05-12-leanvec-lvq/README.md`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod index;
pub mod lvq;
pub mod projection;

pub use index::{FlatIndex, LeanVecIndex, LvqIndex, Neighbor, VectorIndex};
pub use lvq::{LvqCode, LvqCodebook};
pub use projection::Projection;
