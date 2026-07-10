//! Partition-index variants: [`BaselineIvf`], [`RandomSpillIvf`], [`SoarIvf`].
//!
//! All three implement [`PartitionIndex`], enabling swap-in comparison at the
//! benchmark harness level. Vectors are `f32` slices; the index owns the full
//! dataset row-major, indexed by `u32` id.

use crate::kmeans::{kmeans_pp, KMeansConfig, KMeansModel};
use crate::vec_math::l2_sq;

/// Common trait for IVF-family partition indexes with optional duplicate spill.
pub trait PartitionIndex {
    /// Build the index from row-major `data` of shape `(n, dim)`, using
    /// `k` centroids trained by k-means++.
    ///
    /// # Panics
    /// Panics if `data.len() % dim != 0`, or if `data` is empty.
    fn build(data: Vec<f32>, dim: usize, k: usize, seed: u64) -> Self
    where
        Self: Sized;

    /// Return `top_k` nearest neighbors to `query` by probing `nprobe`
    /// nearest partitions. Results are sorted by ascending squared distance.
    fn search(&self, query: &[f32], nprobe: usize, top_k: usize) -> Vec<SearchResult>;

    /// Descriptive statistics: memory and posting-list layout.
    fn stats(&self) -> PartitionStats;

    /// A short human-readable name for reports.
    fn name(&self) -> &'static str;
}

/// A single search hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// Vector id (row index into the original data).
    pub id: u32,
    /// Squared L2 distance to the query.
    pub distance: f32,
}

/// Storage/layout statistics for a built index.
#[derive(Debug, Clone)]
pub struct PartitionStats {
    /// Number of centroids.
    pub k: usize,
    /// Number of unique points.
    pub n_unique: usize,
    /// Total posting-list entries (n_unique + duplicates).
    pub n_entries: usize,
    /// Ratio `n_entries / n_unique` — 1.0 for baseline, ≤2.0 for spill variants.
    pub duplication_ratio: f32,
    /// Approximate bytes used by posting lists (excluding raw data).
    pub posting_bytes: usize,
    /// Approximate bytes used by centroids.
    pub centroid_bytes: usize,
    /// Total bytes (raw data + centroids + posting lists).
    pub total_bytes: usize,
}

// -----------------------------------------------------------------------------
// Shared internal state
// -----------------------------------------------------------------------------

struct IvfCore {
    data: Vec<f32>,
    dim: usize,
    n: usize,
    model: KMeansModel,
    /// One posting list per centroid, each a list of vector ids.
    lists: Vec<Vec<u32>>,
}

impl IvfCore {
    fn train(data: Vec<f32>, dim: usize, k: usize, seed: u64) -> Self {
        assert!(!data.is_empty(), "IvfCore: empty data");
        assert_eq!(data.len() % dim, 0, "IvfCore: data length not multiple of dim");
        let n = data.len() / dim;
        let cfg = KMeansConfig { k, iterations: 15, seed };
        let model = kmeans_pp(&data, dim, &cfg);
        let lists = vec![Vec::<u32>::new(); model.k];
        Self { data, dim, n, model, lists }
    }

    #[inline]
    fn vec(&self, id: u32) -> &[f32] {
        let i = id as usize;
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    fn probe_lists(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut all: Vec<(usize, f32)> =
            (0..self.model.k)
                .map(|i| (i, l2_sq(query, self.model.centroid(i))))
                .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        all.truncate(nprobe.min(self.model.k));
        all.into_iter().map(|(i, _)| i).collect()
    }

    fn scan_and_dedup(
        &self,
        query: &[f32],
        list_ids: &[usize],
        top_k: usize,
    ) -> Vec<SearchResult> {
        // Dedup via boolean bitset (n bits).
        let mut seen = vec![false; self.n];
        let mut heap: Vec<SearchResult> = Vec::with_capacity(top_k + 1);
        for &lid in list_ids {
            for &id in &self.lists[lid] {
                let idx = id as usize;
                if seen[idx] {
                    continue;
                }
                seen[idx] = true;
                let d = l2_sq(query, self.vec(id));
                if heap.len() < top_k {
                    heap.push(SearchResult { id, distance: d });
                    if heap.len() == top_k {
                        heap.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                    }
                } else if d < heap[top_k - 1].distance {
                    heap[top_k - 1] = SearchResult { id, distance: d };
                    heap.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                }
            }
        }
        if heap.len() < top_k {
            heap.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        }
        heap
    }

    fn stats(&self) -> PartitionStats {
        let n_entries: usize = self.lists.iter().map(|l| l.len()).sum();
        let n_unique = self.n;
        let posting_bytes = n_entries * core::mem::size_of::<u32>();
        let centroid_bytes = self.model.centroids.len() * core::mem::size_of::<f32>();
        let data_bytes = self.data.len() * core::mem::size_of::<f32>();
        PartitionStats {
            k: self.model.k,
            n_unique,
            n_entries,
            duplication_ratio: n_entries as f32 / n_unique.max(1) as f32,
            posting_bytes,
            centroid_bytes,
            total_bytes: posting_bytes + centroid_bytes + data_bytes,
        }
    }
}

// -----------------------------------------------------------------------------
// Baseline: hard IVF, no duplicates
// -----------------------------------------------------------------------------

/// Hard IVF baseline — each point in its single nearest partition.
pub struct BaselineIvf {
    core: IvfCore,
}

impl PartitionIndex for BaselineIvf {
    fn build(data: Vec<f32>, dim: usize, k: usize, seed: u64) -> Self {
        let mut core = IvfCore::train(data, dim, k, seed);
        for i in 0..core.n {
            let (c, _) = core.model.nearest(core.vec(i as u32));
            core.lists[c].push(i as u32);
        }
        Self { core }
    }

    fn search(&self, query: &[f32], nprobe: usize, top_k: usize) -> Vec<SearchResult> {
        let probed = self.core.probe_lists(query, nprobe);
        self.core.scan_and_dedup(query, &probed, top_k)
    }

    fn stats(&self) -> PartitionStats {
        self.core.stats()
    }

    fn name(&self) -> &'static str {
        "BaselineIvf"
    }
}

// -----------------------------------------------------------------------------
// SPANN-style isotropic spill: every point in top-2 nearest centroids
// -----------------------------------------------------------------------------

/// Random-secondary spill (SPANN-style). Duplicates every point into its
/// second-nearest centroid regardless of geometry — an isotropic control
/// against which SOAR's anisotropic loss is measured.
pub struct RandomSpillIvf {
    core: IvfCore,
}

impl PartitionIndex for RandomSpillIvf {
    fn build(data: Vec<f32>, dim: usize, k: usize, seed: u64) -> Self {
        let mut core = IvfCore::train(data, dim, k, seed);
        for i in 0..core.n {
            let top = core.model.top_nearest(core.vec(i as u32), 2);
            for (cid, _) in top {
                core.lists[cid].push(i as u32);
            }
        }
        Self { core }
    }

    fn search(&self, query: &[f32], nprobe: usize, top_k: usize) -> Vec<SearchResult> {
        let probed = self.core.probe_lists(query, nprobe);
        self.core.scan_and_dedup(query, &probed, top_k)
    }

    fn stats(&self) -> PartitionStats {
        self.core.stats()
    }

    fn name(&self) -> &'static str {
        "RandomSpillIvf(top2)"
    }
}

// -----------------------------------------------------------------------------
// SOAR: anisotropic spill with orthogonality-amplified residual loss
// -----------------------------------------------------------------------------

/// SOAR-style anisotropic spill.
///
/// For each point `x` with primary centroid `c1`, the secondary centroid is
/// chosen from `top_pool` candidates (excluding `c1`) to minimize
///
/// ```text
/// L(c2) = ||x - c2||^2 + (lambda - 1) * <x - c2, r/||r||>^2
/// ```
///
/// where `r = x - c1`. `lambda = 1.0` degenerates to plain second-nearest.
/// The paper uses `lambda ≈ 3` for typical vector-search corpora.
pub struct SoarIvf {
    core: IvfCore,
    /// Regularization parameter — how strongly to penalize collinear duplicates.
    pub lambda: f32,
    /// Candidate pool size for secondary selection (defaults to k).
    pub top_pool: usize,
}

impl SoarIvf {
    /// Build with explicit `lambda`. Convenience for benchmarks that sweep λ.
    pub fn build_with_lambda(
        data: Vec<f32>,
        dim: usize,
        k: usize,
        seed: u64,
        lambda: f32,
    ) -> Self {
        let mut core = IvfCore::train(data, dim, k, seed);
        let top_pool = core.model.k;
        let mut residual = vec![0.0f32; dim];
        let mut delta = vec![0.0f32; dim];
        for i in 0..core.n {
            let xv = core.vec(i as u32);
            // Primary: nearest centroid.
            let (c1_id, _) = core.model.nearest(xv);
            // Residual r = x - c1.
            {
                let c1v = core.model.centroid(c1_id);
                for j in 0..dim {
                    residual[j] = xv[j] - c1v[j];
                }
            }
            let r_norm_sq = residual.iter().map(|v| v * v).sum::<f32>();
            let r_norm = r_norm_sq.sqrt();

            // Secondary via SOAR loss.
            let mut best_c2 = usize::MAX;
            let mut best_loss = f32::INFINITY;
            for c2_id in 0..core.model.k {
                if c2_id == c1_id {
                    continue;
                }
                let c2v = core.model.centroid(c2_id);
                for j in 0..dim {
                    delta[j] = xv[j] - c2v[j];
                }
                let d_sq: f32 = delta.iter().map(|v| v * v).sum();
                let parallel = if r_norm > 1e-12 {
                    let mut dot_dr = 0.0f32;
                    for j in 0..dim {
                        dot_dr += delta[j] * residual[j];
                    }
                    let p = dot_dr / r_norm;
                    p * p
                } else {
                    0.0
                };
                let loss = d_sq + (lambda - 1.0) * parallel;
                if loss < best_loss {
                    best_loss = loss;
                    best_c2 = c2_id;
                }
            }

            core.lists[c1_id].push(i as u32);
            if best_c2 != usize::MAX && best_c2 != c1_id {
                core.lists[best_c2].push(i as u32);
            }
        }
        Self { core, lambda, top_pool }
    }
}

impl PartitionIndex for SoarIvf {
    fn build(data: Vec<f32>, dim: usize, k: usize, seed: u64) -> Self {
        // Default lambda = 3.0, matching the SOAR paper's typical sweep midpoint.
        Self::build_with_lambda(data, dim, k, seed, 3.0)
    }

    fn search(&self, query: &[f32], nprobe: usize, top_k: usize) -> Vec<SearchResult> {
        let probed = self.core.probe_lists(query, nprobe);
        self.core.scan_and_dedup(query, &probed, top_k)
    }

    fn stats(&self) -> PartitionStats {
        self.core.stats()
    }

    fn name(&self) -> &'static str {
        "SoarIvf(λ=3.0)"
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Xorshift64;

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = Xorshift64::new(seed);
        (0..n * dim).map(|_| rng.next_signed_f32()).collect()
    }

    #[test]
    fn baseline_builds_and_searches() {
        let data = synth(200, 8, 1);
        let idx = BaselineIvf::build(data.clone(), 8, 4, 42);
        let q = &data[0..8];
        let hits = idx.search(q, 4, 5);
        assert!(!hits.is_empty());
        // The point at index 0 should be a top-1 (or near-top) hit when nprobe=k.
        assert_eq!(hits[0].id, 0);
        assert!(hits[0].distance <= 1e-4);
    }

    #[test]
    fn baseline_stats_have_no_duplicates() {
        let data = synth(200, 8, 1);
        let idx = BaselineIvf::build(data, 8, 4, 42);
        let s = idx.stats();
        assert_eq!(s.n_entries, s.n_unique);
        assert!((s.duplication_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn random_spill_doubles_entries() {
        let data = synth(200, 8, 1);
        let idx = RandomSpillIvf::build(data, 8, 4, 42);
        let s = idx.stats();
        assert_eq!(s.n_entries, 2 * s.n_unique);
        assert!((s.duplication_ratio - 2.0).abs() < 1e-6);
    }

    #[test]
    fn soar_doubles_entries_and_names_lambda() {
        let data = synth(200, 8, 1);
        let idx = SoarIvf::build(data, 8, 4, 42);
        let s = idx.stats();
        assert_eq!(s.n_entries, 2 * s.n_unique);
        assert!(idx.name().contains("Soar"));
    }

    #[test]
    fn soar_lambda1_matches_random_spill_by_count() {
        // With lambda=1, SOAR should pick second-nearest centroid every time,
        // which is exactly what RandomSpillIvf does.
        let data = synth(300, 8, 1);
        let soar = SoarIvf::build_with_lambda(data.clone(), 8, 5, 42, 1.0);
        let rand = RandomSpillIvf::build(data, 8, 5, 42);
        // Same total entries and duplication ratio.
        assert_eq!(soar.stats().n_entries, rand.stats().n_entries);
    }

    #[test]
    fn search_top_k_is_sorted_ascending() {
        let data = synth(500, 16, 7);
        let idx = SoarIvf::build(data.clone(), 16, 8, 99);
        let q = &data[0..16];
        let hits = idx.search(q, 4, 10);
        for w in hits.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }
}
