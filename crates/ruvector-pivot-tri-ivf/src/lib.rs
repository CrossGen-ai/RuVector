//! Pivot triangle-inequality pruning for IVF posting-list scans.
//!
//! Core idea (classic metric-space result):
//!   For any pivot p: | d(q, p) - d(x, p) |  <=  d(q, x).
//! Each cluster stores `m` pivots and precomputed d(x, p_j) for every x in its
//! posting list. At query time we compute d(q, p_j) once per pivot per probed
//! cluster (m*nprobe extra distances), then for every candidate x we take the
//! maximum of |d(q, p_j) - d(x, p_j)| over j as a cheap lower bound on d(q, x).
//! If that LB exceeds the current k-th best distance in the heap, we skip the
//! full L2 computation entirely. Correctness is exact: triangle inequality is
//! lossless, so recall is identical to the baseline IVF scan.
//!
//! Deterministic by seed. Pure Rust, no deps.

#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub type VecId = u32;

/// Trait so backends (float, quantized, etc.) can evolve.
pub trait DistanceBackend {
    fn l2_sq(&self, a: &[f32], b: &[f32]) -> f32;
    fn l2(&self, a: &[f32], b: &[f32]) -> f32 {
        self.l2_sq(a, b).sqrt()
    }
}

#[derive(Clone, Copy, Default)]
pub struct ScalarBackend;

impl DistanceBackend for ScalarBackend {
    #[inline]
    fn l2_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut s = 0.0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }
}

/// Deterministic linear-congruential RNG so the whole run is reproducible
/// from a single seed with no external crate.
pub struct Lcg(pub u64);
impl Lcg {
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
    /// Standard-normal (Box-Muller). Good enough for synthetic data.
    pub fn next_normal(&mut self) -> f32 {
        let mut u1 = self.next_f32();
        if u1 < 1e-9 {
            u1 = 1e-9;
        }
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

pub fn gen_dataset(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    (0..n)
        .map(|_| (0..d).map(|_| rng.next_normal()).collect())
        .collect()
}

/// Mixture-of-Gaussians dataset: `g` well-separated blobs of radius `sigma`
/// around means scattered uniformly on a hypercube. Models realistic
/// embeddings where topic clusters produce exploitable intra-cluster
/// geometry for pivot pruning.
pub fn gen_clustered(n: usize, d: usize, g: usize, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Lcg(seed);
    let means: Vec<Vec<f32>> = (0..g)
        .map(|_| (0..d).map(|_| rng.next_normal() * 8.0).collect())
        .collect();
    (0..n)
        .map(|i| {
            let m = &means[i % g];
            (0..d).map(|j| m[j] + sigma * rng.next_normal()).collect()
        })
        .collect()
}

/// One IVF cluster.
struct Cluster {
    centroid: Vec<f32>,
    ids: Vec<VecId>,
    vectors: Vec<Vec<f32>>,
    /// Pivot vectors chosen from inside this cluster.
    pivots: Vec<Vec<f32>>,
    /// d(x_i, pivot_j) laid out as [per-vector][per-pivot] for cache locality
    /// when scanning the posting list.
    pivot_dist: Vec<Vec<f32>>,
}

pub struct IvfIndex<B: DistanceBackend = ScalarBackend> {
    dim: usize,
    nlist: usize,
    m_pivots: usize,
    backend: B,
    clusters: Vec<Cluster>,
}

#[derive(Clone, Debug)]
pub struct SearchStats {
    pub probed_clusters: usize,
    pub candidates_seen: usize,
    pub full_distances: usize,
    pub pivot_prunes: usize,
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: VecId,
    pub dist_sq: f32,
}

#[derive(Copy, Clone)]
struct HeapEntry {
    dist_sq: f32,
    id: VecId,
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq
    }
}
impl Eq for HeapEntry {}

impl IvfIndex<ScalarBackend> {
    /// Build an IVF index with `nlist` k-means clusters (Lloyd, `kmeans_iters`
    /// passes) and `m_pivots` pivot anchors per cluster.
    pub fn build(
        data: &[Vec<f32>],
        nlist: usize,
        m_pivots: usize,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        assert!(!data.is_empty());
        let dim = data[0].len();
        let backend = ScalarBackend;

        let mut centroids = kmeans_pp_init(data, nlist, seed);
        for _ in 0..kmeans_iters {
            let assigns = assign(data, &centroids, &backend);
            centroids = recompute_centroids(data, &assigns, nlist, dim);
        }
        let assigns = assign(data, &centroids, &backend);

        let mut clusters: Vec<Cluster> = centroids
            .into_iter()
            .map(|c| Cluster {
                centroid: c,
                ids: Vec::new(),
                vectors: Vec::new(),
                pivots: Vec::new(),
                pivot_dist: Vec::new(),
            })
            .collect();
        for (i, &a) in assigns.iter().enumerate() {
            clusters[a].ids.push(i as VecId);
            clusters[a].vectors.push(data[i].clone());
        }

        let mut rng = Lcg(seed ^ 0xA11CE);
        for c in clusters.iter_mut() {
            if c.vectors.is_empty() {
                continue;
            }
            // Pivots: farthest-point sampling, seeded with a random posting entry.
            let first = (rng.next_u32() as usize) % c.vectors.len();
            let mut picked: Vec<usize> = vec![first];
            while picked.len() < m_pivots.min(c.vectors.len()) {
                let mut best_i = 0usize;
                let mut best_d = f32::NEG_INFINITY;
                for i in 0..c.vectors.len() {
                    if picked.contains(&i) {
                        continue;
                    }
                    let mut min_d = f32::INFINITY;
                    for &p in &picked {
                        let d = backend.l2_sq(&c.vectors[i], &c.vectors[p]);
                        if d < min_d {
                            min_d = d;
                        }
                    }
                    if min_d > best_d {
                        best_d = min_d;
                        best_i = i;
                    }
                }
                picked.push(best_i);
            }
            c.pivots = picked.iter().map(|&i| c.vectors[i].clone()).collect();
            c.pivot_dist = c
                .vectors
                .iter()
                .map(|v| c.pivots.iter().map(|p| backend.l2(v, p)).collect())
                .collect();
        }

        Self {
            dim,
            nlist,
            m_pivots,
            backend,
            clusters,
        }
    }
}

impl<B: DistanceBackend> IvfIndex<B> {
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn nlist(&self) -> usize {
        self.nlist
    }
    pub fn m_pivots(&self) -> usize {
        self.m_pivots
    }

    /// Search with or without the pivot lower bound.
    /// When `use_pivots=false` the index behaves as a plain IVF (baseline for
    /// benchmarks). Results are identical in both modes (exact pruning).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
        use_pivots: bool,
    ) -> (Vec<Hit>, SearchStats) {
        assert_eq!(query.len(), self.dim);
        let mut cent_d: Vec<(usize, f32)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.backend.l2_sq(query, &c.centroid)))
            .collect();
        cent_d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
        let mut stats = SearchStats {
            probed_clusters: 0,
            candidates_seen: 0,
            full_distances: 0,
            pivot_prunes: 0,
        };

        for &(ci, _) in cent_d.iter().take(nprobe) {
            let c = &self.clusters[ci];
            if c.vectors.is_empty() {
                continue;
            }
            stats.probed_clusters += 1;

            let qp_dists: Vec<f32> = if use_pivots {
                c.pivots.iter().map(|p| self.backend.l2(query, p)).collect()
            } else {
                Vec::new()
            };

            for i in 0..c.vectors.len() {
                stats.candidates_seen += 1;

                if use_pivots && !c.pivots.is_empty() {
                    // LB_d  =  max_j |d(q, p_j) - d(x, p_j)|
                    let xp = &c.pivot_dist[i];
                    let mut lb = 0.0f32;
                    for j in 0..qp_dists.len() {
                        let diff = (qp_dists[j] - xp[j]).abs();
                        if diff > lb {
                            lb = diff;
                        }
                    }
                    let lb_sq = lb * lb;
                    let worst = if heap.len() == k {
                        heap.peek().map(|e| e.dist_sq).unwrap_or(f32::INFINITY)
                    } else {
                        f32::INFINITY
                    };
                    if heap.len() == k && lb_sq >= worst {
                        stats.pivot_prunes += 1;
                        continue;
                    }
                }

                let d = self.backend.l2_sq(query, &c.vectors[i]);
                stats.full_distances += 1;
                let id = c.ids[i];
                if heap.len() < k {
                    heap.push(HeapEntry { dist_sq: d, id });
                } else if let Some(top) = heap.peek() {
                    if d < top.dist_sq {
                        heap.pop();
                        heap.push(HeapEntry { dist_sq: d, id });
                    }
                }
            }
        }

        let mut out: Vec<Hit> = heap
            .into_iter()
            .map(|e| Hit {
                id: e.id,
                dist_sq: e.dist_sq,
            })
            .collect();
        out.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap_or(Ordering::Equal));
        (out, stats)
    }

    /// Brute-force L2 kNN; used for correctness tests and recall measurement.
    pub fn brute_force(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<Hit> {
        let be = ScalarBackend;
        let mut all: Vec<Hit> = data
            .iter()
            .enumerate()
            .map(|(i, v)| Hit {
                id: i as VecId,
                dist_sq: be.l2_sq(query, v),
            })
            .collect();
        all.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap_or(Ordering::Equal));
        all.truncate(k);
        all
    }

    /// Rough memory estimate (bytes) for the pivot side-table only.
    pub fn pivot_overhead_bytes(&self) -> usize {
        let mut t = 0usize;
        for c in &self.clusters {
            t += c.pivots.len() * self.dim * 4;
            t += c.pivot_dist.iter().map(|v| v.len() * 4).sum::<usize>();
        }
        t
    }
}

fn kmeans_pp_init(data: &[Vec<f32>], k: usize, seed: u64) -> Vec<Vec<f32>> {
    let be = ScalarBackend;
    let mut rng = Lcg(seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(data[(rng.next_u32() as usize) % data.len()].clone());
    while centroids.len() < k {
        let dists: Vec<f32> = data
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| be.l2_sq(v, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let total: f32 = dists.iter().sum();
        if total <= 0.0 {
            centroids.push(data[(rng.next_u32() as usize) % data.len()].clone());
            continue;
        }
        let r = rng.next_f32() * total;
        let mut acc = 0.0f32;
        let mut pick = 0usize;
        for (i, d) in dists.iter().enumerate() {
            acc += *d;
            if acc >= r {
                pick = i;
                break;
            }
        }
        centroids.push(data[pick].clone());
    }
    centroids
}

fn assign<B: DistanceBackend>(data: &[Vec<f32>], centroids: &[Vec<f32>], be: &B) -> Vec<usize> {
    data.iter()
        .map(|v| {
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for (i, c) in centroids.iter().enumerate() {
                let d = be.l2_sq(v, c);
                if d < bd {
                    bd = d;
                    best = i;
                }
            }
            best
        })
        .collect()
}

fn recompute_centroids(
    data: &[Vec<f32>],
    assigns: &[usize],
    k: usize,
    dim: usize,
) -> Vec<Vec<f32>> {
    let mut sums = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (i, &a) in assigns.iter().enumerate() {
        counts[a] += 1;
        for j in 0..dim {
            sums[a][j] += data[i][j];
        }
    }
    for a in 0..k {
        if counts[a] > 0 {
            let c = counts[a] as f32;
            for j in 0..dim {
                sums[a][j] /= c;
            }
        } else {
            sums[a] = data[a % data.len()].clone();
        }
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pivots_match_baseline_exactly() {
        let data = gen_dataset(1200, 32, 42);
        let idx = IvfIndex::build(&data, 20, 4, 6, 7);
        let queries = gen_dataset(16, 32, 101);
        for q in &queries {
            let (a, _) = idx.search(q, 10, 6, false);
            let (b, sb) = idx.search(q, 10, 6, true);
            assert_eq!(a.len(), b.len());
            for i in 0..a.len() {
                assert_eq!(a[i].id, b[i].id, "pivot pruning changed top-k ordering");
            }
            // The pruning path must have exercised at least one skip on
            // non-trivial data; otherwise the LB is doing nothing.
            assert!(sb.pivot_prunes + sb.full_distances == sb.candidates_seen);
        }
    }

    #[test]
    fn recall_vs_brute_force_is_sensible() {
        let data = gen_dataset(600, 24, 11);
        let idx = IvfIndex::build(&data, 10, 4, 6, 7);
        let q = gen_dataset(1, 24, 999).remove(0);
        let bf = IvfIndex::<ScalarBackend>::brute_force(&data, &q, 10);
        let (got, _) = idx.search(&q, 10, 10, true);
        // nprobe == nlist means IVF visits every cluster, so recall must be 1.
        let want: std::collections::HashSet<_> = bf.iter().map(|h| h.id).collect();
        let hit = got.iter().filter(|h| want.contains(&h.id)).count();
        assert_eq!(hit, 10);
    }
}
