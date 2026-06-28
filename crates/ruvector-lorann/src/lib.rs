//! ruvector-lorann — LoRANN: Low-Rank Approximate Nearest Neighbor Search
//!
//! Implements the NeurIPS 2024 LoRANN method (Jääsaari, Hyvönen, Roos):
//! per-cluster reduced-rank regression that approximates the query→database
//! inner products `q^T x_i` as `(q^T V_c) (V_c^T x_i)` with a per-cluster
//! orthonormal basis `V_c ∈ R^{d×r}`, `r ≪ d`. After IVF-style cluster pruning,
//! a candidate cluster of size `n_c` scores all members in `O(n_c · r)` flops
//! instead of `O(n_c · d)`, then a small top-`m` set is rescored exactly.
//!
//! ## Backend trait — `InnerProductIndex`
//!
//! Three swappable backends ship in this crate:
//!   * `BruteForceIndex` — exact `O(nd)` baseline (the SOTA-validation honesty knob).
//!   * `IvfIndex` — k-means + nprobe full-dim scoring (`O(K d) + O(p n_c d)`).
//!   * `LoRannIndex` — k-means + per-cluster rank-`r` regression + exact rerank.
//!
//! All vectors are L2-normalized at insert/query time so inner-product top-k
//! coincides with cosine top-k (the regime LoRANN was designed for).
//!
//! ## What this crate is NOT
//!
//! * Not a vector compression scheme — full vectors are kept (rerank needs them).
//!   For compression, see `ruvector-leanvec` / `ruvector-anisotropic-pq`.
//! * Not a graph index — orthogonal to HNSW; can be combined as a re-scorer
//!   (future work, see research doc).
//!
//! ## Safety / determinism
//!
//! The deterministic LCG below makes `train()` reproducible from a seed; no
//! `rand` crate dependency is pulled in. All math is `f32` host code, no SIMD
//! intrinsics — SIMD/SoA layout is left to the production crate (see ADR-272).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod kmeans;
pub mod lowrank;
pub mod index;

pub use index::{
    BruteForceIndex, IvfIndex, LoRannConfig, LoRannIndex, IvfConfig, InnerProductIndex, Neighbor,
};

/// Tiny deterministic LCG so this crate is `rand`-free and reproducible.
#[derive(Debug, Clone, Copy)]
pub struct Lcg(pub u64);

impl Lcg {
    /// Construct from seed. Seed 0 is remapped to a non-degenerate state.
    pub fn new(seed: u64) -> Self {
        Lcg(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    /// Next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    /// Uniform `[0,1)` `f32`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Uniform `usize` in `[0, n)`.
    pub fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n.max(1)
    }
}

/// L2-normalize a vector in place. Zero vectors are left zero.
pub fn l2_normalize(v: &mut [f32]) {
    let mut s = 0.0f32;
    for &x in v.iter() {
        s += x * x;
    }
    if s > 0.0 {
        let inv = 1.0 / s.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Inner product of two equal-length slices. `debug_assert`s the lengths match.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot: dim mismatch");
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod smoke {
    use super::*;

    #[test]
    fn lcg_is_deterministic() {
        let mut a = Lcg::new(42);
        let mut b = Lcg::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn l2_normalize_unit_norm() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_stays_zero() {
        let mut v = vec![0.0_f32; 4];
        l2_normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
