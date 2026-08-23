//! ruvector-soar — SOAR (Spilling with Orthogonality-Amplified Residuals)
//!
//! Nightly research crate for ADR-339. Implements three swappable IVF backends
//! that share a common `PartitionIndex` trait so backends can be evolved
//! independently:
//!
//! * [`IvfTop1`]         — baseline: assign each vector to the nearest centroid.
//! * [`IvfNaiveSpill`]   — spilling: assign to the two nearest centroids.
//! * [`IvfSoar`]         — SOAR: pick the second partition using an
//!                          orthogonality-amplified loss
//!                          `L(c') = ||x-c'||² + λ · ⟨x-c₁, x-c'⟩² / ||x-c₁||²`
//!                          (Sun et al., NeurIPS 2024, Google Research).
//!
//! Scoring at query time is exact (L2 on assigned bucket contents) so we
//! isolate the effect of the *assignment strategy* on recall.
//!
//! Everything is `#![deny(missing_docs)]`-friendly, no `unsafe`, no external
//! deps. Deterministic — driven by a xorshift PRNG passed explicitly.

#![deny(unsafe_code)]

pub mod kmeans;
pub mod rng;

use core::cmp::Ordering;

/// A single vector with its original id.
#[derive(Clone, Debug)]
pub struct Vector {
    /// 1-based id kept for eval / debugging.
    pub id: u32,
    /// Feature values.
    pub data: Vec<f32>,
}

/// Squared L2 distance.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// Small util — top-k on a growing max-heap-of-negatives (simple O(n log k)).
fn topk(mut scored: Vec<(f32, u32)>, k: usize) -> Vec<(f32, u32)> {
    scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Assignment plan for a single vector across partitions.
#[derive(Clone, Debug)]
pub struct Assignment {
    /// (partition_id, primary=0 / secondary=1)
    pub buckets: Vec<u32>,
}

/// The public contract every backend obeys.
pub trait PartitionIndex {
    /// Backend name (for reporting).
    fn name(&self) -> &'static str;
    /// Number of partitions.
    fn nlist(&self) -> usize;
    /// Bytes of postings storage (ids only, f32=4 per centroid slot ignored).
    fn posting_bytes(&self) -> usize;
    /// Nearest-neighbor top-k search visiting `nprobe` partitions.
    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(f32, u32)>;
}

/// Materialized IVF core shared by all three backends.
pub struct IvfCore {
    #[allow(dead_code)]
    dim: usize,
    nlist: usize,
    centroids: Vec<Vec<f32>>,
    /// vectors indexed by id
    vecs: Vec<Vector>,
    /// posting lists: for each partition, list of vector ids
    postings: Vec<Vec<u32>>,
}

impl IvfCore {
    /// Rank partitions by squared L2 from a query.
    fn probe_order(&self, query: &[f32]) -> Vec<(f32, u32)> {
        let mut order: Vec<(f32, u32)> = (0..self.nlist)
            .map(|c| (sq_l2(query, &self.centroids[c]), c as u32))
            .collect();
        order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        order
    }

    fn search_inner(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(f32, u32)> {
        let probes = self.probe_order(query);
        let mut seen = vec![false; self.vecs.len()];
        let mut scored: Vec<(f32, u32)> = Vec::with_capacity(nprobe * 32);
        let nprobe = nprobe.min(self.nlist);
        for i in 0..nprobe {
            let cid = probes[i].1 as usize;
            for &vid in &self.postings[cid] {
                let vu = vid as usize;
                if seen[vu] {
                    continue;
                }
                seen[vu] = true;
                let d = sq_l2(query, &self.vecs[vu].data);
                scored.push((d, vid));
            }
        }
        topk(scored, k)
    }

    fn posting_bytes_inner(&self) -> usize {
        self.postings.iter().map(|p| p.len() * 4).sum()
    }
}

/// Baseline IVF (top-1 assignment).
pub struct IvfTop1 {
    core: IvfCore,
}

impl IvfTop1 {
    /// Build from raw vectors and pretrained centroids.
    pub fn build(vecs: Vec<Vector>, centroids: Vec<Vec<f32>>) -> Self {
        let dim = centroids[0].len();
        let nlist = centroids.len();
        let mut postings = vec![Vec::<u32>::new(); nlist];
        for v in &vecs {
            let mut best = (f32::INFINITY, 0usize);
            for (c, cen) in centroids.iter().enumerate() {
                let d = sq_l2(&v.data, cen);
                if d < best.0 {
                    best = (d, c);
                }
            }
            postings[best.1].push(v.id);
        }
        Self { core: IvfCore { dim, nlist, centroids, vecs, postings } }
    }
}

impl PartitionIndex for IvfTop1 {
    fn name(&self) -> &'static str { "IvfTop1" }
    fn nlist(&self) -> usize { self.core.nlist }
    fn posting_bytes(&self) -> usize { self.core.posting_bytes_inner() }
    fn search(&self, q: &[f32], k: usize, nprobe: usize) -> Vec<(f32, u32)> {
        self.core.search_inner(q, k, nprobe)
    }
}

/// Naive spilling — assign each vector to the top-2 centroids.
pub struct IvfNaiveSpill {
    core: IvfCore,
}

impl IvfNaiveSpill {
    /// Build spilling to `spill` nearest partitions per vector (spill >= 1).
    pub fn build(vecs: Vec<Vector>, centroids: Vec<Vec<f32>>, spill: usize) -> Self {
        let dim = centroids[0].len();
        let nlist = centroids.len();
        let mut postings = vec![Vec::<u32>::new(); nlist];
        for v in &vecs {
            let mut all: Vec<(f32, u32)> = centroids
                .iter()
                .enumerate()
                .map(|(c, cen)| (sq_l2(&v.data, cen), c as u32))
                .collect();
            all.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            for j in 0..spill.min(nlist) {
                postings[all[j].1 as usize].push(v.id);
            }
        }
        Self { core: IvfCore { dim, nlist, centroids, vecs, postings } }
    }
}

impl PartitionIndex for IvfNaiveSpill {
    fn name(&self) -> &'static str { "IvfNaiveSpill" }
    fn nlist(&self) -> usize { self.core.nlist }
    fn posting_bytes(&self) -> usize { self.core.posting_bytes_inner() }
    fn search(&self, q: &[f32], k: usize, nprobe: usize) -> Vec<(f32, u32)> {
        self.core.search_inner(q, k, nprobe)
    }
}

/// SOAR — Spilling with Orthogonality-Amplified Residuals.
///
/// The secondary partition `c'` is chosen to minimise
///
/// ```text
///   L_soar(c') = ||x - c'||²  +  λ · ⟨x - c₁, x - c'⟩² / ||x - c₁||²
/// ```
///
/// where `c₁` is the nearest (primary) centroid. When `λ = 0` this reduces to
/// [`IvfNaiveSpill`] with `spill=2`. The paper recommends λ ≈ 1.5.
pub struct IvfSoar {
    core: IvfCore,
    /// Lambda parameter used at build time (kept for reporting).
    pub lambda: f32,
}

impl IvfSoar {
    /// Build a SOAR IVF (primary + one SOAR-selected secondary per vector).
    pub fn build(vecs: Vec<Vector>, centroids: Vec<Vec<f32>>, lambda: f32) -> Self {
        let dim = centroids[0].len();
        let nlist = centroids.len();
        let mut postings = vec![Vec::<u32>::new(); nlist];
        for v in &vecs {
            // Find primary c₁.
            let mut best = (f32::INFINITY, 0usize);
            for (c, cen) in centroids.iter().enumerate() {
                let d = sq_l2(&v.data, cen);
                if d < best.0 { best = (d, c); }
            }
            let c1 = best.1;
            postings[c1].push(v.id);

            // Residual r1 = x - c1
            let dim = v.data.len();
            let mut r1 = vec![0f32; dim];
            for i in 0..dim { r1[i] = v.data[i] - centroids[c1][i]; }
            let r1_sq = dot(&r1, &r1).max(1e-12);

            // Score every other centroid with SOAR loss; pick argmin.
            let mut best2 = (f32::INFINITY, usize::MAX);
            for c in 0..nlist {
                if c == c1 { continue; }
                let mut r2 = vec![0f32; dim];
                for i in 0..dim { r2[i] = v.data[i] - centroids[c][i]; }
                let d2 = dot(&r2, &r2);
                let ip = dot(&r1, &r2);
                let l = d2 + lambda * (ip * ip) / r1_sq;
                if l < best2.0 { best2 = (l, c); }
            }
            if best2.1 != usize::MAX {
                postings[best2.1].push(v.id);
            }
        }
        Self { core: IvfCore { dim, nlist, centroids, vecs, postings }, lambda }
    }
}

impl PartitionIndex for IvfSoar {
    fn name(&self) -> &'static str { "IvfSoar" }
    fn nlist(&self) -> usize { self.core.nlist }
    fn posting_bytes(&self) -> usize { self.core.posting_bytes_inner() }
    fn search(&self, q: &[f32], k: usize, nprobe: usize) -> Vec<(f32, u32)> {
        self.core.search_inner(q, k, nprobe)
    }
}

/// Ground-truth exact NN (used for recall computation).
pub fn brute_force(vecs: &[Vector], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = vecs
        .iter()
        .map(|v| (sq_l2(query, &v.data), v.id))
        .collect();
    scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(_, id)| id).collect()
}

/// Recall@k for a single query result.
pub fn recall_at_k(gt: &[u32], got: &[(f32, u32)]) -> f32 {
    if gt.is_empty() { return 0.0; }
    let mut hit = 0usize;
    for &id in gt {
        if got.iter().any(|(_, g)| *g == id) { hit += 1; }
    }
    hit as f32 / gt.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xor64;

    fn tiny_dataset() -> (Vec<Vector>, Vec<Vec<f32>>) {
        // 40 vectors in 8-d, 4 well-separated clusters.
        let mut rng = Xor64::new(0xDEAD_BEEFu64);
        let mut vecs = Vec::new();
        let centers: Vec<[f32; 8]> = (0..4)
            .map(|c| {
                let mut a = [0f32; 8];
                for x in &mut a { *x = ((c as f32) * 3.0) + rng.gauss() * 0.05; }
                a
            })
            .collect();
        for c in 0..4 {
            for i in 0..10 {
                let mut v = vec![0f32; 8];
                for k in 0..8 { v[k] = centers[c][k] + rng.gauss() * 0.15; }
                vecs.push(Vector { id: (c * 10 + i) as u32, data: v });
            }
        }
        let cents: Vec<Vec<f32>> = centers.iter().map(|a| a.to_vec()).collect();
        (vecs, cents)
    }

    #[test]
    fn top1_finds_true_neighbor_in_own_bucket() {
        let (vecs, cents) = tiny_dataset();
        let ix = IvfTop1::build(vecs.clone(), cents.clone());
        let q = &vecs[3].data;
        let r = ix.search(q, 1, 1);
        assert_eq!(r[0].1, vecs[3].id);
    }

    #[test]
    fn soar_lambda_zero_matches_naive_spill_size() {
        let (vecs, cents) = tiny_dataset();
        let a = IvfNaiveSpill::build(vecs.clone(), cents.clone(), 2);
        let b = IvfSoar::build(vecs, cents, 0.0);
        // Same total number of postings (2N entries).
        assert_eq!(a.posting_bytes(), b.posting_bytes());
    }

    #[test]
    fn soar_never_reduces_recall_vs_top1_on_toy() {
        let (vecs, cents) = tiny_dataset();
        let base = IvfTop1::build(vecs.clone(), cents.clone());
        let soar = IvfSoar::build(vecs.clone(), cents.clone(), 1.5);
        let mut base_hits = 0f32;
        let mut soar_hits = 0f32;
        for v in &vecs {
            let gt = brute_force(&vecs, &v.data, 5);
            base_hits += recall_at_k(&gt, &base.search(&v.data, 5, 1));
            soar_hits += recall_at_k(&gt, &soar.search(&v.data, 5, 1));
        }
        // SOAR (nprobe=1) should meet or exceed baseline with same probe budget
        // because the vector itself is guaranteed placed in the primary bucket.
        assert!(soar_hits >= base_hits - 1e-4, "soar={soar_hits} base={base_hits}");
    }
}
