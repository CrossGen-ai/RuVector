//! Portable software prefetch helper.
//!
//! On x86_64 we emit a `PREFETCHT0` instruction targeting the L1 data
//! cache (locality = 3 in the LLVM/Rust intrinsic model). On aarch64 we
//! emit a `PRFM PLDL1KEEP` via the corresponding intrinsic. On other
//! architectures the call collapses to a `black_box` read hint that
//! prevents dead-code elimination but performs no useful work.
//!
//! All prefetches are read-only, non-temporal-hint free. The helper is
//! `#[inline(always)]` so its cost in release mode is a single machine
//! instruction — measured on the benchmark host at <1 ns amortised.

#![allow(unused_imports)]

use core::hint::black_box;

/// Prefetch a cache line into L1 for reading.
///
/// Safe: prefetching an invalid address is a no-op on all supported ISAs
/// per Intel SDM and Arm ARM. The pointer is never dereferenced.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // Manually emit `prfm pldl1keep, [x]`. Stable Rust does not yet
        // expose a portable prefetch intrinsic on aarch64.
        core::arch::asm!(
            "prfm pldl1keep, [{x}]",
            x = in(reg) ptr,
            options(readonly, nostack, preserves_flags)
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = black_box(ptr);
    }
}

/// Prefetch every cache line touched by an f32 vector of `dim` lanes.
///
/// A single 64-byte cache line holds 16 f32 lanes, so we issue
/// `ceil(dim / 16)` prefetches. For dim=128 that is 8 prefetches, well
/// below the ~10-in-flight limit of a modern L1D miss buffer.
#[inline(always)]
pub fn prefetch_vector(vec: &[f32]) {
    let ptr = vec.as_ptr();
    let lines = vec.len().div_ceil(16);
    for l in 0..lines {
        // SAFETY: we do not dereference; we only compute an address that
        // is within the allocation (prefetch of the last line is
        // guaranteed in-bounds because `lines = ceil(len/16)`).
        let p = unsafe { ptr.add(l * 16) };
        prefetch_read(p);
    }
}

/// Policy controlling how many neighbours ahead we prefetch.
#[derive(Copy, Clone, Debug)]
pub enum PrefetchPolicy {
    /// No prefetch — baseline.
    None,
    /// Prefetch exactly the next `k` neighbours while computing the
    /// current distance.
    Fixed(usize),
    /// Start at `initial`, grow by one when the last neighbour was
    /// accepted into the candidate heap (evidence prefetch paid off),
    /// shrink by one when it was rejected. Clamped to `[1, max]`.
    Adaptive { initial: usize, max: usize },
}

impl PrefetchPolicy {
    #[inline]
    pub fn initial_depth(&self) -> usize {
        match *self {
            PrefetchPolicy::None => 0,
            PrefetchPolicy::Fixed(k) => k,
            PrefetchPolicy::Adaptive { initial, .. } => initial,
        }
    }
    #[inline]
    pub fn max_depth(&self) -> usize {
        match *self {
            PrefetchPolicy::None => 0,
            PrefetchPolicy::Fixed(k) => k,
            PrefetchPolicy::Adaptive { max, .. } => max,
        }
    }
    #[inline]
    pub fn is_adaptive(&self) -> bool {
        matches!(self, PrefetchPolicy::Adaptive { .. })
    }
}
