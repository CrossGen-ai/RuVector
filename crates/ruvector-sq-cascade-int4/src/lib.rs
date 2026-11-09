//! Cascaded int8 + int4-residual scalar quantization.
//!
//! Encode pipeline per vector:
//!   1. Compute global per-dim mean/scale from training set.
//!   2. int8 base: round((x - mean) / scale8).
//!   3. reconstruct base: mean + q8 * scale8.
//!   4. residual = x - base; per-dim residual scale from training.
//!   5. int4 residual: round(residual / scale4).
//!   6. storage: int8 code (D bytes) + packed int4 (D/2 bytes) = 1.5 * D bytes.
//!
//! Effective precision ≈ log2(256 * 16) = 12 bits for a 1.5x int8 footprint.

pub mod quant;
pub mod dataset;
pub mod eval;

pub use quant::{Quantizer, Fp32Baseline, Int8Sq, CascadeInt8Int4, QuantStats};
pub use dataset::{gaussian_dataset, uniform_queries};
pub use eval::{recall_at_k, EvalReport};
