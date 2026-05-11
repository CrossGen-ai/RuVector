//! IVF index with optional SOAR secondary assignment.
//!
//! Three assignment modes:
//! - [`Assignment::Naive`]: classic IVF — each point in one cell.
//! - [`Assignment::IsotropicSpillover`]: each point in 2 cells, second is just
//!   the second-nearest centroid (λ = 0, no anisotropic term).
//! - [`Assignment::SoarAnisotropic { lambda }`]: SOAR — secondary centroid
//!   minimizes ||x - c||² + λ · (<x - c, r₁>/||r₁||)² where r₁ is the primary
//!   residual. λ=1.0 matches the value reported in the NeurIPS 2023 paper.

use crate::distance::{dot, l2_sq, residual};
use crate::{Result, SoarError};

#[derive(Clone, Copy, Debug)]
pub enum Assignment {
    Naive,
    IsotropicSpillover,
    SoarAnisotropic { lambda: f32 },
}

#[derive(Clone, Debug)]
pub struct SoarConfig {
    pub assignment: Assignment,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: u32,
    pub distance: f32,
}

pub struct IvfIndex {
    dim: usize,
    n_lists: usize,
    centroids: Vec<f32>,     // n_lists * dim
    vectors: Vec<f32>,       // n * dim, owned (PoC keeps full vectors for rerank)
    n: usize,
    /// For each cell, the IDs of vectors assigned to it.
    posting: Vec<Vec<u32>>,
    assignment: Assignment,
}

impl IvfIndex {
    /// Build an IVF index given pre-trained centroids and dataset vectors.
    pub fn build(
        centroids: Vec<f32>,
        n_lists: usize,
        dim: usize,
        vectors: Vec<f32>,
        cfg: &SoarConfig,
    ) -> Result<Self> {
        if vectors.is_empty() {
            return Err(SoarError::EmptyDataset);
        }
        if vectors.len() % dim != 0 {
            return Err(SoarError::DimensionMismatch {
                expected: dim,
                got: vectors.len(),
            });
        }
        if centroids.len() != n_lists * dim {
            return Err(SoarError::DimensionMismatch {
                expected: n_lists * dim,
                got: centroids.len(),
            });
        }
        let n = vectors.len() / dim;
        let mut posting: Vec<Vec<u32>> = (0..n_lists).map(|_| Vec::new()).collect();
        let mut resid_buf = vec![0.0f32; dim];

        for i in 0..n {
            let x = &vectors[i * dim..(i + 1) * dim];

            // Primary: nearest centroid.
            let (primary, _) = nearest_centroid(x, &centroids, n_lists, dim);
            posting[primary].push(i as u32);

            // Secondary (if enabled).
            match cfg.assignment {
                Assignment::Naive => { /* nothing */ }
                Assignment::IsotropicSpillover => {
                    let secondary = second_nearest_centroid(
                        x,
                        &centroids,
                        n_lists,
                        dim,
                        primary,
                    );
                    posting[secondary].push(i as u32);
                }
                Assignment::SoarAnisotropic { lambda } => {
                    let pc = &centroids[primary * dim..(primary + 1) * dim];
                    residual(x, pc, &mut resid_buf);
                    let r_norm_sq = dot(&resid_buf, &resid_buf);
                    let inv_r_norm_sq = if r_norm_sq > 1e-12 {
                        1.0 / r_norm_sq
                    } else {
                        0.0
                    };
                    let secondary = soar_secondary(
                        x,
                        &resid_buf,
                        &centroids,
                        n_lists,
                        dim,
                        primary,
                        lambda,
                        inv_r_norm_sq,
                    );
                    posting[secondary].push(i as u32);
                }
            }
        }

        Ok(Self {
            dim,
            n_lists,
            centroids,
            vectors,
            n,
            posting,
            assignment: cfg.assignment,
        })
    }

    pub fn n(&self) -> usize { self.n }
    pub fn dim(&self) -> usize { self.dim }
    pub fn n_lists(&self) -> usize { self.n_lists }
    pub fn assignment(&self) -> Assignment { self.assignment }

    /// Total bytes used by the index excluding raw vectors (centroids + postings).
    pub fn index_overhead_bytes(&self) -> usize {
        self.centroids.len() * 4
            + self.posting.iter().map(|p| p.len() * 4 + 24).sum::<usize>()
    }

    /// Total bytes including raw vectors (held by the index).
    pub fn total_bytes(&self) -> usize {
        self.vectors.len() * 4 + self.index_overhead_bytes()
    }

    /// Search query: probe `nprobe` nearest cells, dedupe candidate IDs,
    /// exact-rerank, return top-k.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim);
        let nprobe = nprobe.min(self.n_lists);

        // Rank centroids by distance to query.
        let mut cell_scores: Vec<(usize, f32)> = (0..self.n_lists)
            .map(|c| {
                let cc = &self.centroids[c * self.dim..(c + 1) * self.dim];
                (c, l2_sq(query, cc))
            })
            .collect();
        cell_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Collect unique candidate IDs from the top-nprobe cells.
        // A point appears in <=2 cells with spillover; a bitset dedupes cheaply.
        let mut seen = vec![false; self.n];
        let mut cands: Vec<u32> = Vec::with_capacity(self.n / self.n_lists * nprobe + 16);
        for (cell, _) in cell_scores.iter().take(nprobe) {
            for &id in &self.posting[*cell] {
                let u = id as usize;
                if !seen[u] {
                    seen[u] = true;
                    cands.push(id);
                }
            }
        }

        // Exact rerank.
        let mut scored: Vec<SearchResult> = cands
            .into_iter()
            .map(|id| {
                let v = &self.vectors[(id as usize) * self.dim..((id as usize) + 1) * self.dim];
                SearchResult { id, distance: l2_sq(query, v) }
            })
            .collect();
        scored.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        scored.truncate(k);
        scored
    }
}

fn nearest_centroid(
    x: &[f32],
    centroids: &[f32],
    n_lists: usize,
    dim: usize,
) -> (usize, f32) {
    let mut best = 0usize;
    let mut bestd = f32::INFINITY;
    for c in 0..n_lists {
        let cc = &centroids[c * dim..(c + 1) * dim];
        let dd = l2_sq(x, cc);
        if dd < bestd {
            bestd = dd;
            best = c;
        }
    }
    (best, bestd)
}

fn second_nearest_centroid(
    x: &[f32],
    centroids: &[f32],
    n_lists: usize,
    dim: usize,
    exclude: usize,
) -> usize {
    let mut best = if exclude == 0 { 1 } else { 0 };
    let mut bestd = f32::INFINITY;
    for c in 0..n_lists {
        if c == exclude { continue; }
        let cc = &centroids[c * dim..(c + 1) * dim];
        let dd = l2_sq(x, cc);
        if dd < bestd {
            bestd = dd;
            best = c;
        }
    }
    best
}

/// SOAR secondary centroid selection.
///
/// Minimizes `L(c) = ||x - c||² + λ · (<x - c, r₁>)² / ||r₁||²` over c ≠ primary.
///
/// The penalty term is large when `x - c` is parallel to `r₁ = x - c_primary`,
/// pushing the secondary residual to be orthogonal to the primary residual.
/// Geometrically: if the query's actual nearest neighbor sits far from
/// `c_primary`, the *direction* it sits in is captured by `r₁`; picking a
/// secondary cell whose residual is perpendicular to `r₁` covers the
/// orthogonal complement, so the union of two cells reaches more of the space
/// at fixed nprobe than picking two nearby centroids.
#[allow(clippy::too_many_arguments)]
fn soar_secondary(
    x: &[f32],
    r1: &[f32],
    centroids: &[f32],
    n_lists: usize,
    dim: usize,
    primary: usize,
    lambda: f32,
    inv_r_norm_sq: f32,
) -> usize {
    let mut best = if primary == 0 { 1 } else { 0 };
    let mut bestl = f32::INFINITY;
    // Avoid an extra allocation: compute <x-c, r1> = <x, r1> - <c, r1>
    let x_dot_r1 = dot(x, r1);
    for c in 0..n_lists {
        if c == primary { continue; }
        let cc = &centroids[c * dim..(c + 1) * dim];
        let dist_sq = l2_sq(x, cc);
        let proj = x_dot_r1 - dot(cc, r1); // <x - c, r1>
        let penalty = proj * proj * inv_r_norm_sq;
        let loss = dist_sq + lambda * penalty;
        if loss < bestl {
            bestl = loss;
            best = c;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::{kmeans_lloyd, KMeansConfig};
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn synthetic(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut r = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| r.gen::<f32>() * 2.0 - 1.0).collect()
    }

    fn brute_top_k(data: &[f32], n: usize, d: usize, q: &[f32], k: usize) -> Vec<u32> {
        let mut s: Vec<(u32, f32)> = (0..n)
            .map(|i| (i as u32, l2_sq(&data[i * d..(i + 1) * d], q)))
            .collect();
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        s.into_iter().take(k).map(|(id, _)| id).collect()
    }

    #[test]
    fn build_searches_naive() {
        let n = 500; let d = 16; let k = 10;
        let data = synthetic(n, d, 1);
        let cents = kmeans_lloyd(&data, n, d,
            &KMeansConfig { k: 16, iters: 10, seed: 2 });
        let idx = IvfIndex::build(
            cents, 16, d, data.clone(),
            &SoarConfig { assignment: Assignment::Naive },
        ).unwrap();
        // probe everything → exact
        let q = synthetic(1, d, 99);
        let got = idx.search(&q, k, 16);
        let truth = brute_top_k(&data, n, d, &q, k);
        let got_ids: Vec<u32> = got.iter().map(|r| r.id).collect();
        assert_eq!(got_ids, truth);
    }

    #[test]
    fn soar_at_least_as_good_as_naive_low_nprobe() {
        // SOAR should match-or-beat naive recall at low nprobe (the whole point).
        let n = 800; let d = 32; let k = 10; let nprobe = 2; let nlists = 32;
        let data = synthetic(n, d, 3);
        let qs = synthetic(40, d, 4);
        let cents = kmeans_lloyd(&data, n, d,
            &KMeansConfig { k: nlists, iters: 12, seed: 5 });

        let naive = IvfIndex::build(
            cents.clone(), nlists, d, data.clone(),
            &SoarConfig { assignment: Assignment::Naive },
        ).unwrap();
        let soar = IvfIndex::build(
            cents, nlists, d, data.clone(),
            &SoarConfig { assignment: Assignment::SoarAnisotropic { lambda: 1.0 } },
        ).unwrap();

        let (mut hits_naive, mut hits_soar) = (0, 0);
        let mut total = 0;
        for qi in 0..40 {
            let q = &qs[qi * d..(qi + 1) * d];
            let truth = brute_top_k(&data, n, d, q, k);
            let nv: Vec<u32> = naive.search(q, k, nprobe).iter().map(|r| r.id).collect();
            let sv: Vec<u32> = soar.search(q, k, nprobe).iter().map(|r| r.id).collect();
            for t in &truth {
                if nv.contains(t) { hits_naive += 1; }
                if sv.contains(t) { hits_soar += 1; }
                total += 1;
            }
        }
        let rn = hits_naive as f32 / total as f32;
        let rs = hits_soar as f32 / total as f32;
        // Allow tiny tolerance for ties on synthetic data.
        assert!(rs + 0.01 >= rn,
            "expected SOAR recall >= naive recall, got soar={} naive={}", rs, rn);
    }
}
