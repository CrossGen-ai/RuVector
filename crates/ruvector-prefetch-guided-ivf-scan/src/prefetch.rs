//! Portable software-prefetch primitives.
//!
//! Stable Rust as of 1.89 exposes `core::arch::x86_64::_mm_prefetch` on x86_64
//! and stable inline assembly on aarch64 (Apple Silicon, Graviton, etc.).
//! Other targets get a no-op fallback that the optimizer can elide.
//!
//! The design goal is a single `prefetch_read(ptr)` call site that lowers to
//! one `PREFETCHT0` on x86_64 or one `PRFM PLDL1KEEP` on aarch64. We avoid
//! nightly `std::intrinsics::prefetch_*` and the `prefetch` crate so the
//! benchmark stays inside stable-toolchain reach for any RuVector contributor.

/// Prefetch temporal locality hints. Kept minimal — only what the IVF scan
/// benchmark actually consumes. Extend as needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locality {
    /// Data will be reused imminently — target L1 (`_MM_HINT_T0` / `PLDL1KEEP`).
    L1,
    /// Data will be reused soon but not next — target L2 (`_MM_HINT_T1` / `PLDL2KEEP`).
    L2,
}

/// Issue a software prefetch for `ptr`. Safe wrapper: the pointer is only
/// dereferenced by the CPU's prefetcher, never by our code, so a garbage
/// pointer produces at worst a wasted prefetch, never a segfault. The wrapper
/// still requires the caller to hand over a real pointer they intend to read.
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T, locality: Locality) {
    // SAFETY: prefetch instructions are architecturally defined to never fault,
    // never observably read, and never modify state. The `unsafe` blocks below
    // wrap either an intrinsic (x86_64) or inline assembly (aarch64) that
    // matches that guarantee. The fallback is a pure no-op.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0, _MM_HINT_T1};
        let hint = match locality {
            Locality::L1 => _MM_HINT_T0,
            Locality::L2 => _MM_HINT_T1,
        };
        _mm_prefetch(ptr as *const i8, hint);
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        // PRFM prfop, [Xn]. prfop = <type><target><policy>:
        //   type = PLD (prefetch for load)
        //   target = L1 or L2
        //   policy = KEEP (retain in cache)
        match locality {
            Locality::L1 => {
                core::arch::asm!(
                    "prfm pldl1keep, [{p}]",
                    p = in(reg) ptr,
                    options(nostack, preserves_flags, readonly),
                );
            }
            Locality::L2 => {
                core::arch::asm!(
                    "prfm pldl2keep, [{p}]",
                    p = in(reg) ptr,
                    options(nostack, preserves_flags, readonly),
                );
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
        let _ = locality;
    }
}

/// Convenience: prefetch a whole `[f32]` slice header's first cache line.
#[inline(always)]
pub fn prefetch_slice_l1(slice: &[f32]) {
    if !slice.is_empty() {
        prefetch_read(slice.as_ptr(), Locality::L1);
    }
}

/// Compute a reasonable adaptive lookahead in *elements* (not bytes) given the
/// per-vector byte size and an estimated DRAM latency budget in cache lines.
///
/// Heuristic: we want to stay `budget_lines * 64` bytes ahead of the read
/// cursor, so lookahead ≈ ceil((budget_lines * 64) / bytes_per_vec).
/// Clamped to `[1, 32]` — beyond 32 the reorder buffer stops helping and we
/// begin polluting L2 with vectors we may never touch.
#[inline]
pub fn adaptive_lookahead(bytes_per_vec: usize, budget_cache_lines: usize) -> usize {
    if bytes_per_vec == 0 {
        return 1;
    }
    let bytes = budget_cache_lines.saturating_mul(64);
    let raw = (bytes + bytes_per_vec - 1) / bytes_per_vec;
    raw.clamp(1, 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefetch_null_is_safe_no_panic() {
        // A null pointer prefetch is architecturally a no-op. If this test ever
        // segfaults we've introduced a real dereference by mistake.
        prefetch_read::<f32>(core::ptr::null(), Locality::L1);
        prefetch_read::<f32>(core::ptr::null(), Locality::L2);
    }

    #[test]
    fn prefetch_valid_slice_no_panic() {
        let v = vec![1.0f32; 128];
        prefetch_slice_l1(&v);
        prefetch_read(v.as_ptr().wrapping_add(64), Locality::L2);
    }

    #[test]
    fn adaptive_lookahead_scales_with_vec_size() {
        // 512-byte vectors (128 f32) with a 4-cache-line budget → 1 ahead.
        assert_eq!(adaptive_lookahead(512, 4), 1);
        // 128-byte vectors (32 f32) with a 4-cache-line budget → 2 ahead.
        assert_eq!(adaptive_lookahead(128, 4), 2);
        // Tiny vectors (32 bytes = 8 f32) with 4-line budget → 8 ahead.
        assert_eq!(adaptive_lookahead(32, 4), 8);
        // Clamp lower.
        assert_eq!(adaptive_lookahead(1_000_000, 1), 1);
        // Clamp upper.
        assert_eq!(adaptive_lookahead(1, 1_000_000), 32);
    }
}
