//! ruvector-symphonyqg
//!
//! A unified small-world graph + RaBitQ-style 1-bit quantization index.
//!
//! Core ideas (inspired by SymphonyQG, SIGMOD 2024, Gou et al.):
//!   1. Apply a random orthogonal-ish rotation (Walsh–Hadamard sign flips +
//!      random permutation) to break anisotropy and concentrate L2 mass.
//!   2. Encode each rotated vector as a 1-bit-per-dimension sign code, plus a
//!      scalar L2 norm. Distance between query and database vector under this
//!      code is a calibrated function of popcount(q_bits XOR x_bits).
//!   3. Build a small-world NSW-style graph whose `M` neighbors per node are
//!      chosen using FULL-PRECISION distance (offline cost), but graph
//!      *traversal at query time uses the quantized distance* for all hops.
//!   4. Re-rank the top `ef` candidates with full-precision distance to
//!      recover recall.
//!
//! This is a self-contained PoC. No external graph or BLAS dependencies, no
//! mocks, no TODOs. Real benchmarks live in `src/main.rs` and `benches/`.

pub mod quantizer;
pub mod graph;
pub mod index;

pub use index::{SymphonyQg, SymphonyQgParams};
pub use quantizer::RaBitQuantizer;
