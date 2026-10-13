//! Rotation-Ensemble RaBitQ.
//!
//! Classical RaBitQ (Gao & Long, SIGMOD 2024) rotates a vector by a single
//! random orthogonal matrix `R`, stores `sign(R x)` as `D` bits, and
//! estimates inner products via `(1/√D) · <sign(R x), R q>`. Under a Haar-
//! uniform `R` the estimator is unbiased with variance `O(‖x‖²‖q‖²/D)`.
//!
//! If you want tighter estimates there are two orthogonal knobs:
//!
//! 1. **More bits per rotation** — "extended" / multibit RaBitQ (b bits per
//!    coordinate, uniform/non-uniform codebook). Error shrinks like
//!    `2^{-b}` per coordinate.
//! 2. **More independent rotations** — this crate. Draw `K` independent
//!    rotations `R_1,…,R_K`, store `sign(R_k x)` for each, and average the
//!    `K` single-rotation estimators at query time. Variance shrinks like
//!    `1/K` because the single-rotation errors are asymptotically
//!    uncorrelated (independent Haar rotations ⇒ independent rotated
//!    coordinates).
//!
//! Storage is identical for the two knobs when b = K: both pay `K` bits per
//! coordinate per vector. The question this crate answers empirically is
//! which knob is more recall-efficient at a fixed storage budget.
//!
//! ## What this crate ships
//!
//! * [`RabitqSingle`] — baseline 1-bit RaBitQ with a single rotation.
//! * [`RabitqEnsemble`] — K independent rotations, averaged.
//! * [`FlatF32Index`] — exact top-k, for the recall gold standard.
//! * Three rotation backends ([`RotationKind`]):
//!   `HaarDense` (true Haar via Gram–Schmidt),
//!   `HadamardSigned` (D₁·H·D₂·H·D₃, O(D log D)),
//!   `HadamardSingle` (D·H, O(D log D), one sign vector).
//!
//! Everything is `#![forbid(unsafe_code)]`, no external BLAS, deterministic
//! given `(dim, seed, kind, data)`.
//!
//! ## Example
//!
//! ```no_run
//! use ruvector_rabitq_ensemble::{RabitqEnsemble, RotationKind, Index};
//!
//! let data: Vec<Vec<f32>> = (0..10_000)
//!     .map(|i| (0..128).map(|j| ((i * 31 + j) as f32).sin()).collect())
//!     .collect();
//! let idx = RabitqEnsemble::build(&data, 4, RotationKind::HadamardSigned, 0xC0FFEE);
//! let q = vec![0.1_f32; 128];
//! let top10 = idx.search(&q, 10);
//! println!("top id = {}", top10[0].id);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod flat;
pub mod index;
pub mod rotation;
pub mod single;
pub mod ensemble;

pub use error::EnsembleError;
pub use flat::FlatF32Index;
pub use index::{Index, SearchResult};
pub use rotation::{Rotation, RotationKind};
pub use single::RabitqSingle;
pub use ensemble::RabitqEnsemble;
