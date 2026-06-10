//! ruvector-adaptive-probe
//!
//! Adaptive nprobe IVF index. The contribution is a `ProbeStrategy` trait that
//! decides, *during* an IVF query, whether to stop scanning further inverted
//! lists. The baseline visits a fixed nprobe; the adaptive strategies inspect
//! the running top-k score distribution and the cluster-centroid lower bound to
//! cut the probe budget when continuing cannot plausibly improve recall@k.
//!
//! Three strategies are shipped (all implement `ProbeStrategy`):
//!
//! 1. [`FixedNprobe`] — classical IVF, visits exactly `nprobe` lists.
//! 2. [`PlateauProbe`] — stops when the top-k score has not improved for
//!    `patience` consecutive probes (score-plateau detection).
//! 3. [`MarginBudget`] — stops when the centroid-to-query distance of the next
//!    list exceeds the current k-th score by `margin` (provable miss bound:
//!    no point in that list can be closer than its centroid minus the cluster
//!    radius). With `radius=0` this is a pure lower-bound prune.
//!
//! All distances are squared-L2. Vectors are dense `f32` slices stored
//! contiguously. The index is rebuildable, swappable, and small enough to
//! audit in one sitting.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::cmp::Ordering;
use thiserror::Error;

pub mod dataset;
pub mod strategies;

pub use strategies::{FixedNprobe, MarginBudget, PlateauProbe};

#[derive(Debug, Error)]
pub enum AdaptiveProbeError {
    #[error("dimension mismatch: index has {expected}, query has {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("nlist must be > 0")]
    InvalidNlist,
    #[error("k must be > 0")]
    InvalidK,
}

/// Squared L2 distance. Unrolled to give the autovectorizer something to chew
/// on; on x86_64 with `-C target-cpu=native` this lowers to AVX2/AVX-512.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    let n = a.len();
    let chunks = n / 4;
    for i in 0..chunks {
        let j = i * 4;
        let d0 = a[j] - b[j];
        let d1 = a[j + 1] - b[j + 1];
        let d2 = a[j + 2] - b[j + 2];
        let d3 = a[j + 3] - b[j + 3];
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for i in (chunks * 4)..n {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// A point id in the corpus.
pub type Id = u32;

/// Strategy interface. Implementations are stateless across queries; per-query
/// scratch lives in [`ProbeState`].
pub trait ProbeStrategy: Send + Sync {
    fn name(&self) -> &'static str;
    /// Construct fresh per-query state.
    fn new_state(&self) -> ProbeState;
    /// Called after each list is scanned with the up-to-date top-k.
    /// Returns `true` to keep probing, `false` to stop.
    fn should_continue(
        &self,
        state: &mut ProbeState,
        probes_done: usize,
        nlist: usize,
        next_centroid_sqd: Option<f32>,
        topk: &TopK,
    ) -> bool;
}

#[derive(Default)]
pub struct ProbeState {
    pub last_best: f32,
    pub stagnant: usize,
}

/// A bounded max-heap top-k tracker (sorted small array; k is tiny in practice).
pub struct TopK {
    k: usize,
    items: Vec<(f32, Id)>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }

    /// Returns the current k-th smallest distance, or `f32::INFINITY` if not yet full.
    #[inline]
    pub fn kth(&self) -> f32 {
        if self.items.len() < self.k {
            f32::INFINITY
        } else {
            self.items[self.k - 1].0
        }
    }

    #[inline]
    pub fn best(&self) -> f32 {
        self.items.first().map(|x| x.0).unwrap_or(f32::INFINITY)
    }

    pub fn push(&mut self, dist: f32, id: Id) {
        if self.items.len() < self.k {
            self.items.push((dist, id));
            self.items
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            return;
        }
        if dist >= self.kth() {
            return;
        }
        // Replace worst then bubble down.
        self.items[self.k - 1] = (dist, id);
        let mut i = self.k - 1;
        while i > 0 && self.items[i].0 < self.items[i - 1].0 {
            self.items.swap(i, i - 1);
            i -= 1;
        }
    }

    pub fn into_sorted(self) -> Vec<(f32, Id)> {
        self.items
    }
}

/// An IVF index. Centroids are computed by k-means|| seeding + a few Lloyd
/// iterations on a sample. Inverted lists hold raw vectors so distance
/// computation does not need a separate gather.
pub struct IvfIndex {
    dim: usize,
    nlist: usize,
    centroids: Vec<f32>, // [nlist * dim]
    /// For each list: ids of assigned points, flattened, plus offsets.
    list_ids: Vec<Vec<Id>>,
    /// For each list: contiguous f32 store of the points, [|list| * dim].
    list_vecs: Vec<Vec<f32>>,
}

impl IvfIndex {
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn nlist(&self) -> usize {
        self.nlist
    }
    pub fn n(&self) -> usize {
        self.list_ids.iter().map(|l| l.len()).sum()
    }

    pub fn memory_bytes(&self) -> usize {
        let c = self.centroids.len() * std::mem::size_of::<f32>();
        let v: usize = self.list_vecs.iter().map(|v| v.len() * 4).sum();
        let i: usize = self.list_ids.iter().map(|l| l.len() * 4).sum();
        c + v + i
    }

    /// Build an IVF index from a corpus laid out as `[n * dim]`.
    pub fn build(
        corpus: &[f32],
        n: usize,
        dim: usize,
        nlist: usize,
        seed: u64,
    ) -> Result<Self, AdaptiveProbeError> {
        if nlist == 0 {
            return Err(AdaptiveProbeError::InvalidNlist);
        }
        assert_eq!(corpus.len(), n * dim);
        let centroids = kmeans_lloyd(corpus, n, dim, nlist, 8, seed);
        // Assign each point to its nearest centroid.
        let mut list_ids = vec![Vec::<Id>::new(); nlist];
        let mut list_vecs = vec![Vec::<f32>::new(); nlist];
        for i in 0..n {
            let p = &corpus[i * dim..(i + 1) * dim];
            let mut best = (f32::INFINITY, 0usize);
            for c in 0..nlist {
                let cv = &centroids[c * dim..(c + 1) * dim];
                let d = sq_l2(p, cv);
                if d < best.0 {
                    best = (d, c);
                }
            }
            list_ids[best.1].push(i as Id);
            list_vecs[best.1].extend_from_slice(p);
        }
        Ok(Self {
            dim,
            nlist,
            centroids,
            list_ids,
            list_vecs,
        })
    }

    /// Centroid distances for `query`, returned sorted ascending as
    /// `(sq_dist, list_id)`.
    pub fn ranked_centroids(&self, query: &[f32]) -> Vec<(f32, usize)> {
        let mut v = Vec::with_capacity(self.nlist);
        for c in 0..self.nlist {
            let cv = &self.centroids[c * self.dim..(c + 1) * self.dim];
            v.push((sq_l2(query, cv), c));
        }
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        v
    }

    /// Adaptive top-k search. `max_nprobe` caps the strategy; it can stop early.
    /// Returns `(results, probes_used, points_scanned)`.
    pub fn search<S: ProbeStrategy>(
        &self,
        query: &[f32],
        k: usize,
        max_nprobe: usize,
        strategy: &S,
    ) -> Result<(Vec<(f32, Id)>, usize, usize), AdaptiveProbeError> {
        if query.len() != self.dim {
            return Err(AdaptiveProbeError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k == 0 {
            return Err(AdaptiveProbeError::InvalidK);
        }
        let ranked = self.ranked_centroids(query);
        let nprobe = max_nprobe.min(self.nlist);
        let mut topk = TopK::new(k);
        let mut state = strategy.new_state();
        let mut points_scanned = 0usize;
        let mut probes_done = 0usize;
        for slot in 0..nprobe {
            let (_centroid_sqd, list_id) = ranked[slot];
            let ids = &self.list_ids[list_id];
            let vecs = &self.list_vecs[list_id];
            for (off, &id) in ids.iter().enumerate() {
                let v = &vecs[off * self.dim..(off + 1) * self.dim];
                let d = sq_l2(query, v);
                topk.push(d, id);
            }
            points_scanned += ids.len();
            probes_done += 1;
            let next_centroid_sqd = if slot + 1 < nprobe {
                Some(ranked[slot + 1].0)
            } else {
                None
            };
            if !strategy.should_continue(
                &mut state,
                probes_done,
                self.nlist,
                next_centroid_sqd,
                &topk,
            ) {
                break;
            }
        }
        Ok((topk.into_sorted(), probes_done, points_scanned))
    }

    /// Brute-force k-NN — ground truth for recall measurement.
    pub fn brute_force(&self, query: &[f32], k: usize) -> Vec<(f32, Id)> {
        let mut topk = TopK::new(k);
        for c in 0..self.nlist {
            let ids = &self.list_ids[c];
            let vecs = &self.list_vecs[c];
            for (off, &id) in ids.iter().enumerate() {
                let v = &vecs[off * self.dim..(off + 1) * self.dim];
                let d = sq_l2(query, v);
                topk.push(d, id);
            }
        }
        topk.into_sorted()
    }
}

/// Tiny k-means: kmeans-style random init + Lloyd iterations. Not production
/// quality (no k-means++ seeding, no minibatch) — sufficient for IVF buckets
/// at the scales used in our benchmarks (≤ 50k points).
fn kmeans_lloyd(corpus: &[f32], n: usize, dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = vec![0.0f32; k * dim];
    // Random init from corpus points.
    use rand::Rng;
    for c in 0..k {
        let i = rng.gen_range(0..n);
        centroids[c * dim..(c + 1) * dim].copy_from_slice(&corpus[i * dim..(i + 1) * dim]);
    }
    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        // Assign.
        for i in 0..n {
            let p = &corpus[i * dim..(i + 1) * dim];
            let mut best = (f32::INFINITY, 0u32);
            for c in 0..k {
                let cv = &centroids[c * dim..(c + 1) * dim];
                let d = sq_l2(p, cv);
                if d < best.0 {
                    best = (d, c as u32);
                }
            }
            assign[i] = best.1;
        }
        // Update.
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assign[i] as usize;
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += corpus[i * dim + d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed empty cluster from a random point.
                let i = rng.gen_range(0..n);
                centroids[c * dim..(c + 1) * dim].copy_from_slice(&corpus[i * dim..(i + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            }
        }
    }
    centroids
}

/// Compute recall@k of `got` against `truth`.
pub fn recall_at_k(got: &[(f32, Id)], truth: &[(f32, Id)], k: usize) -> f32 {
    let k = k.min(got.len()).min(truth.len());
    if k == 0 {
        return 1.0;
    }
    use std::collections::HashSet;
    let tset: HashSet<Id> = truth.iter().take(k).map(|(_, id)| *id).collect();
    let hits = got.iter().take(k).filter(|(_, id)| tset.contains(id)).count();
    hits as f32 / k as f32
}

/// Build a Gaussian-cluster synthetic dataset for repeatable benchmarks.
pub fn make_synthetic(
    n: usize,
    dim: usize,
    n_clusters: usize,
    sigma: f32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    // Cluster centers spread across [-3, 3].
    let center_normal = Normal::new(0.0f32, 3.0).unwrap();
    let mut centers = vec![0.0f32; n_clusters * dim];
    for v in centers.iter_mut() {
        *v = center_normal.sample(&mut rng);
    }
    let mut data = vec![0.0f32; n * dim];
    use rand::Rng;
    for i in 0..n {
        let c = rng.gen_range(0..n_clusters);
        for d in 0..dim {
            let noise = normal.sample(&mut rng) * sigma;
            data[i * dim + d] = centers[c * dim + d] + noise;
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_l2_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(sq_l2(&a, &a), 0.0);
    }

    #[test]
    fn sq_l2_basic() {
        let a = [0.0f32, 0.0, 0.0, 0.0];
        let b = [1.0f32, 2.0, 2.0, 0.0];
        assert!((sq_l2(&a, &b) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn topk_orders_correctly() {
        let mut t = TopK::new(3);
        for (d, id) in [(5.0, 0u32), (1.0, 1), (4.0, 2), (2.0, 3), (3.0, 4)] {
            t.push(d, id);
        }
        let v = t.into_sorted();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].1, 1);
        assert_eq!(v[1].1, 3);
        assert_eq!(v[2].1, 4);
    }

    #[test]
    fn ivf_build_and_search_recovers_self() {
        let n = 200;
        let dim = 8;
        let data = make_synthetic(n, dim, 4, 0.3, 42);
        let idx = IvfIndex::build(&data, n, dim, 8, 7).unwrap();
        assert_eq!(idx.n(), n);
        // Query equals a known point: best result must be that point at distance 0.
        let q = data[0..dim].to_vec();
        let strat = FixedNprobe::new(8);
        let (res, _probes, _scanned) = idx.search(&q, 5, 8, &strat).unwrap();
        assert_eq!(res[0].1, 0);
        assert!(res[0].0 < 1e-6);
    }

    #[test]
    fn adaptive_stops_before_full_nprobe() {
        let n = 400;
        let dim = 16;
        let data = make_synthetic(n, dim, 6, 0.25, 11);
        let idx = IvfIndex::build(&data, n, dim, 16, 3).unwrap();
        let q = data[0..dim].to_vec();
        let strat = PlateauProbe::new(2);
        let (_res, probes, _scanned) = idx.search(&q, 10, 16, &strat).unwrap();
        // Self-query: top-k saturates immediately, plateau must trigger early stop.
        assert!(probes < 16, "expected early stop, got probes={probes}");
    }
}
