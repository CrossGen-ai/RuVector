//! 4-bit scalar quantization for approximate-nearest-neighbor search.
//!
//! Two coders are shipped behind a common [`Codec`] trait:
//! - [`UniformSQ4`]: linear min/max bins per dimension (classic SQ4).
//! - [`LloydMaxSQ4`]: Lloyd-Max optimal non-uniform bins per dimension.
//!
//! Both store two codes per byte (nibble packing). Asymmetric L2 distance is
//! implemented in [`distance`] using a per-query lookup table so the hot path
//! is pure integer memory accesses.

pub mod distance;
pub mod quant;
pub mod train;

pub use distance::{asym_l2_lut, asym_l2_scan};
pub use quant::{Codec, LloydMaxSQ4, UniformSQ4};
pub use train::fit_lloyd_max_1d;

/// Fixed number of levels for a 4-bit code.
pub const NLEVELS: usize = 16;

/// Pack two u8 values in `[0, 16)` into a byte: low nibble first.
#[inline(always)]
pub fn pack_nibble(a: u8, b: u8) -> u8 {
    debug_assert!(a < 16 && b < 16);
    a | (b << 4)
}

/// Unpack a byte into two u8 values in `[0, 16)`.
#[inline(always)]
pub fn unpack_nibble(byte: u8) -> (u8, u8) {
    (byte & 0x0F, byte >> 4)
}
