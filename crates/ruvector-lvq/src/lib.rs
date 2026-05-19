//! ruvector-lvq: Locally-Adaptive Vector Quantization
//!
//! Based on "Locally-Adaptive Quantization for Streaming Vector Search"
//! (Aguerrebere et al., arXiv:2402.02044 / NeurIPS 2023). Each vector is
//! centered against a global mean, then individually scaled to fit a B-bit
//! uniform grid. Per-vector (scale, bias, sq-norm) are stored alongside the
//! compressed codes so distances can be decoded with one mul + add per
//! component.
//!
//! Two encoders are provided:
//! - [`LvqOne`]: single-level B-bit LVQ (`B = 4` or `8`)
//! - [`LvqTwo`]: two-level LVQ — 8-bit primary + B2-bit residual (`B2 = 4`)
//!
//! A scalar-quantization baseline ([`Sq8`]) is included for comparison.

pub mod error;
pub mod quantizer;
pub mod lvq1;
pub mod lvq2;
pub mod sq;
pub mod distance;
pub mod recall;

pub use error::LvqError;
pub use quantizer::{Quantizer, Encoded, Code};
pub use lvq1::LvqOne;
pub use lvq2::LvqTwo;
pub use sq::Sq8;
pub use distance::Metric;
