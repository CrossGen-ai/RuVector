//! # ruvector-norm-bucket-ivf
//!
//! Norm-sorted cluster buckets for IVF (Inverted-File) vector search.
//!
//! ## The idea
//!
//! Standard IVF probes `nprobe` clusters and computes the full L2 distance
//! to every vector inside them. For a query `q` with current k-NN worst
//! distance `tau`, most of that work is wasted: any candidate `x` with
//!
//! ```text
//!     ( |q| - |x| )^2  >  tau
//! ```
//!
//! cannot possibly enter the top-k, by the reverse Cauchy-Schwarz /
//! triangle inequality on `L2` norms. Sorting each cluster's members by
//! `|x|` lets us binary-search a **contiguous window** of feasible
//! candidates and skip everything outside it — without touching a single
//! coordinate of `x`.
//!
//! Empirically (see `benches/prune_bench.rs`) this prunes 40-80% of
//! per-cluster candidates at Recall@10 = 1.00 on synthetic Gaussian data,
//! and stacks cleanly with partial-L2 early-abandon inside the surviving
//! window.
//!
//! ## Traits
//!
//! - [`ScanBackend`] — pluggable per-cluster scan strategy.
//! - Built-in impls: [`FullScan`], [`NormBucketScan`], [`NormBucketPartialL2Scan`].
//!
//! ## Public API
//!
//! - [`IvfIndex`] — build with `IvfIndex::build(vectors, num_clusters)`.
//! - [`IvfIndex::search`] — top-k search with a chosen `ScanBackend`.
//! - [`IvfIndex::search_with_stats`] — returns pruning statistics.

#![deny(missing_docs)]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Squared-L2 distance between two equal-length vectors.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Squared-L2 with early-abandon: stop once running sum exceeds `bound`.
///
/// Returns `Some(dist_sq)` if `dist_sq <= bound`, else `None`. Bound is
/// applied as an inclusive threshold.
#[inline]
pub fn l2_sq_early_abandon(a: &[f32], b: &[f32], bound: f32) -> Option<f32> {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    // Unroll by 4 to amortise the compare.
    let mut i = 0;
    while i + 4 <= a.len() {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        if s > bound {
            return None;
        }
        i += 4;
    }
    while i < a.len() {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    if s <= bound {
        Some(s)
    } else {
        None
    }
}

/// Norm of a vector.
#[inline]
pub fn norm(v: &[f32]) -> f32 {
    let mut s = 0f32;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// A single IVF cluster: centroid, member vector ids, and (when built with
/// [`IvfIndex::build`]) member vectors sorted by their L2 norm.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Centroid vector.
    pub centroid: Vec<f32>,
    /// Global ids of the members, ordered to match `sorted_norms` /
    /// `sorted_vectors`.
    pub member_ids: Vec<u32>,
    /// Precomputed member norms, ascending.
    pub sorted_norms: Vec<f32>,
    /// Member vectors, in the same order as `sorted_norms`.
    /// Row-major: `sorted_vectors[i * dim .. (i+1) * dim]`.
    pub sorted_vectors: Vec<f32>,
    /// Per-vector dimension.
    pub dim: usize,
}

impl Cluster {
    /// Number of members in this cluster.
    pub fn len(&self) -> usize {
        self.member_ids.len()
    }

    /// True iff the cluster has no members.
    pub fn is_empty(&self) -> bool {
        self.member_ids.is_empty()
    }

    /// Raw slice for the i-th member vector.
    #[inline]
    pub fn vector(&self, i: usize) -> &[f32] {
        &self.sorted_vectors[i * self.dim..(i + 1) * self.dim]
    }
}

/// Statistics collected during a scan, useful for benchmarking.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScanStats {
    /// Total candidates considered (across probed clusters).
    pub candidates: u64,
    /// Candidates for which a full L2 was computed.
    pub full_distances: u64,
    /// Candidates skipped by norm-bucket bounds (no coordinates touched).
    pub skipped_by_norm: u64,
    /// Candidates for which the early-abandon triggered mid-computation.
    pub early_abandoned: u64,
}

/// Pluggable per-cluster scan backend.
pub trait ScanBackend: Sync {
    /// Given a query, its norm, the k-NN heap's current worst squared
    /// distance (or `f32::INFINITY` if heap not yet full), a cluster, and
    /// a callback to record `(id, dist_sq)` for accepted candidates, scan
    /// the cluster and update `stats`.
    fn scan(
        &self,
        query: &[f32],
        query_norm: f32,
        tau_sq: f32,
        cluster: &Cluster,
        push: &mut dyn FnMut(u32, f32),
        stats: &mut ScanStats,
    );

    /// Human-readable name for reporting.
    fn name(&self) -> &'static str;
}

/// Baseline: touch every member, no pruning.
pub struct FullScan;

impl ScanBackend for FullScan {
    fn scan(
        &self,
        query: &[f32],
        _query_norm: f32,
        _tau_sq: f32,
        cluster: &Cluster,
        push: &mut dyn FnMut(u32, f32),
        stats: &mut ScanStats,
    ) {
        for i in 0..cluster.len() {
            stats.candidates += 1;
            stats.full_distances += 1;
            let d = l2_sq(query, cluster.vector(i));
            push(cluster.member_ids[i], d);
        }
    }

    fn name(&self) -> &'static str {
        "full_scan"
    }
}

/// Norm-bucket pruning only (no early-abandon inside the window).
pub struct NormBucketScan;

impl ScanBackend for NormBucketScan {
    fn scan(
        &self,
        query: &[f32],
        query_norm: f32,
        tau_sq: f32,
        cluster: &Cluster,
        push: &mut dyn FnMut(u32, f32),
        stats: &mut ScanStats,
    ) {
        let n = cluster.len();
        stats.candidates += n as u64;
        if n == 0 {
            return;
        }
        // Feasible window: |q_norm - x_norm| <= sqrt(tau_sq).
        let (lo, hi) = if tau_sq.is_finite() {
            let r = tau_sq.sqrt();
            let lo_norm = query_norm - r;
            let hi_norm = query_norm + r;
            let lo = lower_bound(&cluster.sorted_norms, lo_norm);
            let hi = upper_bound(&cluster.sorted_norms, hi_norm);
            (lo, hi)
        } else {
            (0, n)
        };
        stats.skipped_by_norm += (n - (hi - lo)) as u64;
        for i in lo..hi {
            stats.full_distances += 1;
            let d = l2_sq(query, cluster.vector(i));
            push(cluster.member_ids[i], d);
        }
    }

    fn name(&self) -> &'static str {
        "norm_bucket"
    }
}

/// Norm-bucket pruning + partial-L2 early-abandon inside the window.
pub struct NormBucketPartialL2Scan;

impl ScanBackend for NormBucketPartialL2Scan {
    fn scan(
        &self,
        query: &[f32],
        query_norm: f32,
        tau_sq: f32,
        cluster: &Cluster,
        push: &mut dyn FnMut(u32, f32),
        stats: &mut ScanStats,
    ) {
        let n = cluster.len();
        stats.candidates += n as u64;
        if n == 0 {
            return;
        }
        let (lo, hi) = if tau_sq.is_finite() {
            let r = tau_sq.sqrt();
            let lo_norm = query_norm - r;
            let hi_norm = query_norm + r;
            (
                lower_bound(&cluster.sorted_norms, lo_norm),
                upper_bound(&cluster.sorted_norms, hi_norm),
            )
        } else {
            (0, n)
        };
        stats.skipped_by_norm += (n - (hi - lo)) as u64;
        for i in lo..hi {
            match l2_sq_early_abandon(query, cluster.vector(i), tau_sq) {
                Some(d) => {
                    stats.full_distances += 1;
                    push(cluster.member_ids[i], d);
                }
                None => {
                    stats.early_abandoned += 1;
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "norm_bucket+partial_l2"
    }
}

/// Bounded-size min-heap of `(dist_sq, id)` where the root is the *largest*
/// distance (so we can pop the worst when full).
#[derive(Debug)]
struct TopK {
    k: usize,
    // Max-heap by dist_sq.
    heap: Vec<(f32, u32)>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            heap: Vec::with_capacity(k + 1),
        }
    }

    fn worst(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap[0].0
        }
    }

    fn push(&mut self, id: u32, dist_sq: f32) {
        if self.heap.len() < self.k {
            self.heap.push((dist_sq, id));
            self.sift_up(self.heap.len() - 1);
        } else if dist_sq < self.heap[0].0 {
            self.heap[0] = (dist_sq, id);
            self.sift_down(0);
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.heap[i].0 > self.heap[p].0 {
                self.heap.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut largest = i;
            if l < n && self.heap[l].0 > self.heap[largest].0 {
                largest = l;
            }
            if r < n && self.heap[r].0 > self.heap[largest].0 {
                largest = r;
            }
            if largest == i {
                break;
            }
            self.heap.swap(i, largest);
            i = largest;
        }
    }

    fn into_sorted(mut self) -> Vec<(u32, f32)> {
        self.heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.heap.into_iter().map(|(d, i)| (i, d)).collect()
    }
}

/// Full IVF index with norm-sorted cluster buckets.
pub struct IvfIndex {
    /// Per-cluster storage (centroid + members).
    pub clusters: Vec<Cluster>,
    /// Vector dimensionality.
    pub dim: usize,
}

impl IvfIndex {
    /// Build an IVF index by k-means clustering (Lloyd, `iters` iterations)
    /// then bucketing each cluster's members by L2 norm.
    ///
    /// `vectors` is row-major: `n * dim` floats. Deterministic given `seed`.
    pub fn build(vectors: &[f32], dim: usize, num_clusters: usize, iters: usize, seed: u64) -> Self {
        assert!(dim > 0);
        assert!(!vectors.is_empty());
        assert_eq!(vectors.len() % dim, 0);
        let n = vectors.len() / dim;
        assert!(num_clusters > 0 && num_clusters <= n);

        let mut rng = StdRng::seed_from_u64(seed);

        // k-means++ init.
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(num_clusters);
        let first = rng.gen_range(0..n);
        centroids.push(vectors[first * dim..(first + 1) * dim].to_vec());
        let mut min_d2: Vec<f32> = (0..n)
            .map(|i| l2_sq(&vectors[i * dim..(i + 1) * dim], &centroids[0]))
            .collect();
        while centroids.len() < num_clusters {
            let sum: f32 = min_d2.iter().sum();
            let mut target = rng.gen::<f32>() * sum;
            let mut pick = n - 1;
            for (i, &d) in min_d2.iter().enumerate() {
                if target <= d {
                    pick = i;
                    break;
                }
                target -= d;
            }
            let new_c = vectors[pick * dim..(pick + 1) * dim].to_vec();
            for i in 0..n {
                let d = l2_sq(&vectors[i * dim..(i + 1) * dim], &new_c);
                if d < min_d2[i] {
                    min_d2[i] = d;
                }
            }
            centroids.push(new_c);
        }

        // Lloyd iterations.
        let mut assign = vec![0u32; n];
        for _ in 0..iters {
            // Assign
            for i in 0..n {
                let v = &vectors[i * dim..(i + 1) * dim];
                let mut best = 0u32;
                let mut best_d = f32::INFINITY;
                for (c, cent) in centroids.iter().enumerate() {
                    let d = l2_sq(v, cent);
                    if d < best_d {
                        best_d = d;
                        best = c as u32;
                    }
                }
                assign[i] = best;
            }
            // Update
            let mut sums = vec![vec![0f32; dim]; num_clusters];
            let mut counts = vec![0u32; num_clusters];
            for i in 0..n {
                let c = assign[i] as usize;
                let v = &vectors[i * dim..(i + 1) * dim];
                for j in 0..dim {
                    sums[c][j] += v[j];
                }
                counts[c] += 1;
            }
            for c in 0..num_clusters {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for j in 0..dim {
                        centroids[c][j] = sums[c][j] * inv;
                    }
                }
            }
        }

        // Bucket + sort by norm inside each cluster.
        let mut clusters: Vec<Cluster> = centroids
            .into_iter()
            .map(|centroid| Cluster {
                centroid,
                member_ids: Vec::new(),
                sorted_norms: Vec::new(),
                sorted_vectors: Vec::new(),
                dim,
            })
            .collect();
        // Collect per-cluster (id, norm) then sort.
        let mut per_cluster: Vec<Vec<(u32, f32)>> =
            (0..num_clusters).map(|_| Vec::new()).collect();
        for i in 0..n {
            let c = assign[i] as usize;
            let v = &vectors[i * dim..(i + 1) * dim];
            per_cluster[c].push((i as u32, norm(v)));
        }
        for (c, mut pc) in per_cluster.into_iter().enumerate() {
            pc.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let members = pc.len();
            let mut sorted_vectors = Vec::with_capacity(members * dim);
            let mut member_ids = Vec::with_capacity(members);
            let mut sorted_norms = Vec::with_capacity(members);
            for (id, nm) in pc {
                member_ids.push(id);
                sorted_norms.push(nm);
                sorted_vectors.extend_from_slice(&vectors[id as usize * dim..(id as usize + 1) * dim]);
            }
            clusters[c].member_ids = member_ids;
            clusters[c].sorted_norms = sorted_norms;
            clusters[c].sorted_vectors = sorted_vectors;
        }

        IvfIndex { clusters, dim }
    }

    /// Estimated per-index memory: vectors + norms + ids + centroids.
    pub fn estimated_bytes(&self) -> usize {
        let mut b = 0usize;
        for c in &self.clusters {
            b += c.sorted_vectors.len() * 4;
            b += c.sorted_norms.len() * 4;
            b += c.member_ids.len() * 4;
            b += c.centroid.len() * 4;
        }
        b
    }

    /// Return the `nprobe` cluster indices whose centroids are closest to `query`.
    pub fn probe_clusters(&self, query: &[f32], nprobe: usize) -> Vec<usize> {
        let mut d: Vec<(usize, f32)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq(query, &c.centroid)))
            .collect();
        d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        d.truncate(nprobe);
        d.into_iter().map(|(i, _)| i).collect()
    }

    /// Top-k search using the supplied backend.
    pub fn search<B: ScanBackend>(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        backend: &B,
    ) -> Vec<(u32, f32)> {
        self.search_with_stats(query, k, nprobe, backend).0
    }

    /// Top-k search that also returns scan statistics.
    pub fn search_with_stats<B: ScanBackend>(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        backend: &B,
    ) -> (Vec<(u32, f32)>, ScanStats) {
        assert_eq!(query.len(), self.dim);
        let q_norm = norm(query);
        let probes = self.probe_clusters(query, nprobe);
        let mut topk = TopK::new(k);
        let mut stats = ScanStats::default();
        for ci in probes {
            let cluster = &self.clusters[ci];
            let tau = topk.worst();
            let mut push = |id: u32, d: f32| topk.push(id, d);
            backend.scan(query, q_norm, tau, cluster, &mut push, &mut stats);
        }
        (topk.into_sorted(), stats)
    }
}

/// Return the leftmost index `i` such that `arr[i] >= x` (arr sorted ascending).
#[inline]
pub fn lower_bound(arr: &[f32], x: f32) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let m = lo + (hi - lo) / 2;
        if arr[m] < x {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    lo
}

/// Return the leftmost index `i` such that `arr[i] > x` (arr sorted ascending).
#[inline]
pub fn upper_bound(arr: &[f32], x: f32) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let m = lo + (hi - lo) / 2;
        if arr[m] <= x {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    lo
}

/// Deterministic Gaussian dataset generator for tests/benches.
pub fn gaussian_dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    (0..n * dim).map(|_| normal.sample(&mut rng)).collect()
}

/// Deterministic **mixed-scale** dataset: each vector is Gaussian, then
/// scaled by a per-vector factor drawn log-uniformly in `[scale_lo, scale_hi]`.
///
/// This mimics real-world embeddings where norms vary meaningfully across
/// items (e.g. by document length, image energy). Norm-bucket pruning
/// becomes highly effective on such distributions.
pub fn mixed_scale_dataset(
    n: usize,
    dim: usize,
    seed: u64,
    scale_lo: f32,
    scale_hi: f32,
) -> Vec<f32> {
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let mut out = vec![0f32; n * dim];
    let log_lo = scale_lo.ln();
    let log_hi = scale_hi.ln();
    for i in 0..n {
        let s: f32 = (log_lo + (log_hi - log_lo) * rng.gen::<f32>()).exp();
        for j in 0..dim {
            out[i * dim + j] = s * normal.sample(&mut rng);
        }
    }
    out
}

/// Exact brute-force top-k. Ground truth for recall checks.
pub fn brute_force_topk(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let n = vectors.len() / dim;
    let mut d: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, l2_sq(query, &vectors[i * dim..(i + 1) * dim])))
        .collect();
    d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    d.truncate(k);
    d
}

/// Recall@k of `got` vs `truth` (both same length k).
pub fn recall_at_k(got: &[(u32, f32)], truth: &[(u32, f32)]) -> f32 {
    use std::collections::HashSet;
    let t: HashSet<u32> = truth.iter().map(|x| x.0).collect();
    let hit = got.iter().filter(|x| t.contains(&x.0)).count();
    hit as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_dataset() -> (Vec<f32>, usize) {
        // 3D vectors chosen to sit in distinct norm shells.
        let v: Vec<f32> = vec![
            0.1, 0.0, 0.0, // id 0, norm ~0.1
            0.2, 0.1, 0.0, // id 1
            5.0, 5.0, 5.0, // id 2, norm ~8.66
            5.1, 5.0, 5.0, // id 3
            10.0, 0.0, 0.0, // id 4, norm ~10
            -10.0, 0.0, 0.0, // id 5, norm ~10
        ];
        (v, 3)
    }

    #[test]
    fn bounds_are_correct() {
        let arr = vec![0.0, 1.0, 1.0, 2.0, 3.0];
        assert_eq!(lower_bound(&arr, 1.0), 1);
        assert_eq!(upper_bound(&arr, 1.0), 3);
        assert_eq!(lower_bound(&arr, -1.0), 0);
        assert_eq!(upper_bound(&arr, 100.0), 5);
    }

    #[test]
    fn early_abandon_matches_full() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let d = l2_sq(&a, &b);
        assert!((l2_sq_early_abandon(&a, &b, d + 1e-3).unwrap() - d).abs() < 1e-3);
        assert!(l2_sq_early_abandon(&a, &b, d - 1.0).is_none());
    }

    #[test]
    fn norm_bucket_matches_full_scan_on_tiny() {
        let (v, dim) = tiny_dataset();
        let idx = IvfIndex::build(&v, dim, 2, 5, 42);
        let q = vec![0.15, 0.05, 0.0];
        let full = idx.search(&q, 2, 2, &FullScan);
        let bucket = idx.search(&q, 2, 2, &NormBucketScan);
        let combo = idx.search(&q, 2, 2, &NormBucketPartialL2Scan);
        assert_eq!(full, bucket);
        assert_eq!(full, combo);
    }

    #[test]
    fn synthetic_recall_matches_brute() {
        let n = 2000;
        let dim = 16;
        let v = gaussian_dataset(n, dim, 7);
        let q = gaussian_dataset(1, dim, 99);
        let idx = IvfIndex::build(&v, dim, 32, 10, 42);
        let truth = brute_force_topk(&v, dim, &q, 10);
        // Full scan of ALL clusters must equal brute-force truth.
        let got = idx.search(&q, 10, 32, &FullScan);
        assert_eq!(got.len(), 10);
        let r = recall_at_k(&got, &truth);
        assert!((r - 1.0).abs() < 1e-6, "full-scan of all clusters must be exact, got recall={r}");

        // Norm-bucket must also equal brute-force truth (it is provably
        // lossless when tau is honestly maintained).
        let got_bucket = idx.search(&q, 10, 32, &NormBucketScan);
        let rb = recall_at_k(&got_bucket, &truth);
        assert!((rb - 1.0).abs() < 1e-6, "norm-bucket must be lossless when scanning every cluster, got recall={rb}");

        // Combined backend also lossless.
        let got_combo = idx.search(&q, 10, 32, &NormBucketPartialL2Scan);
        let rc = recall_at_k(&got_combo, &truth);
        assert!((rc - 1.0).abs() < 1e-6, "norm-bucket+partial-L2 must be lossless, got recall={rc}");
    }

    #[test]
    fn pruning_actually_prunes() {
        // Mixed-scale dataset: high norm variance => norm-bucket is effective.
        let n = 5000;
        let dim = 32;
        let v = mixed_scale_dataset(n, dim, 3, 0.2, 5.0);
        let q = mixed_scale_dataset(1, dim, 4, 0.2, 5.0);
        let idx = IvfIndex::build(&v, dim, 64, 12, 1);
        let (_, s_full) = idx.search_with_stats(&q, 10, 8, &FullScan);
        let (_, s_bucket) = idx.search_with_stats(&q, 10, 8, &NormBucketScan);
        assert_eq!(s_full.skipped_by_norm, 0);
        assert!(
            s_bucket.skipped_by_norm > 0,
            "norm-bucket must skip some candidates: {:?}",
            s_bucket
        );
        assert!(s_bucket.full_distances < s_full.full_distances);
    }
}
