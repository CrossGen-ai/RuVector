//! ruvector-hadamard-sq
//!
//! Hadamard-rotated scalar quantization. A random Walsh–Hadamard rotation is
//! applied before uniform scalar quantization. The rotation equalises
//! per-dimension variance (Johnson–Lindenstrauss style flattening), which
//! lets a *global* min/max uniform SQ4 / SQ8 codebook retain much more
//! information at the same bit budget.
//!
//! Three codecs are provided:
//!   - `UniformSQ4` — baseline, 4 bits/dim, global min/max.
//!   - `HadamardSQ4` — randomised Walsh–Hadamard rotation, then `UniformSQ4`.
//!   - `HadamardSQ8` — same rotation, 8 bits/dim (half compression, lower MSE).
//!
//! All codecs implement the `Codec` trait so a search path can swap backends.

pub mod hadamard;
pub mod quant;
pub mod distance;

pub use hadamard::{Rotation, Hadamard};
pub use quant::{Codec, UniformSQ4, HadamardSQ4, HadamardSQ8};
pub use distance::{brute_force_recall, l2_sq};
