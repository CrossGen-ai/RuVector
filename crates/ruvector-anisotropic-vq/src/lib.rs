//! Anisotropic Vector Quantization (AVQ) — ScaNN-style loss-aware PQ.
//!
//! Reference: Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar.
//! "Accelerating Large-Scale Inference with Anisotropic Vector Quantization",
//! ICML 2020.
//!
//! Idea: when ranking by inner product (or normalized cosine), the error
//! component of the quantization residual *parallel* to the query direction
//! distorts ranking far more than the *perpendicular* component. The
//! anisotropic loss reweights training:
//!
//! ```text
//! L = eta * ||r_parallel||^2 + ||r_perpendicular||^2
//! ```
//!
//! with eta >= 1 (eta = 1 recovers ordinary symmetric MSE PQ).
//!
//! The data vector itself acts as the reference direction for its own
//! residual decomposition (Lloyd-style training), under the unit-norm
//! assumption that's standard for cosine/MIPS workloads.

pub mod loss;
pub mod pq;
pub mod search;

pub use loss::{decompose_residual, AnisotropicLoss};
pub use pq::{ProductQuantizer, QuantizerKind};
pub use search::{brute_force_topk, recall_at_k, Hit};
