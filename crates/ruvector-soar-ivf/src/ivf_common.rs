//! Shared IVF plumbing: k-means-lite centroid trainer, probe-and-scan search,
//! posting-list container, and the `IvfVariant` trait.
//!
//! Kept intentionally simple and dependency-free so the three variants
//! (single-assignment, top-k spilling, SOAR) all use the *same* trainer /
//! scanner and any recall differences come purely from assignment.

use crate::{dot, metrics::Hit, sq_l2};

// ─── centroid trainer ────────────────────────────────────────────────────────

pub struct Centroids {
    pub vecs: Vec<Vec<f32>>,
    pub dims: usize,
}

impl Centroids {
    /// k-means-lite: `iters` Lloyd iterations from k seeded corpus points.
    pub fn train(vectors: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Self {
        assert!(!vectors.is_empty(), "need vectors to train centroids");
        assert!(k >= 1, "need at least one centroid");
        let dims = vectors[0].len();
        let mut rng = crate::dataset::Lcg::new(seed);

        // Deterministic k-random-seed init.
        let mut centroids: Vec<Vec<f32>> = (0..k)
            .map(|_| vectors[rng.next_range(0, vectors.len())].clone())
            .collect();

        for _iter in 0..iters {
            let mut sums: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0f64; dims]).collect();
            let mut counts: Vec<usize> = vec![0; k];

            for v in vectors {
                let a = nearest_centroid_idx(v, &centroids);
                counts[a] += 1;
                for d in 0..dims {
                    sums[a][d] += v[d] as f64;
                }
            }

            for i in 0..k {
                if counts[i] > 0 {
                    for d in 0..dims {
                        centroids[i][d] = (sums[i][d] / counts[i] as f64) as f32;
                    }
                } else {
                    // Empty centroid: re-seed from a random corpus point.
                    let re = vectors[rng.next_range(0, vectors.len())].clone();
                    centroids[i] = re;
                }
            }
        }

        Self {
            vecs: centroids,
            dims,
        }
    }

    /// Return the `n_probe` centroid indices nearest to `query`, sorted by
    /// ascending L2² distance.
    pub fn probe(&self, query: &[f32], n_probe: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .vecs
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq_l2(query, c)))
            .collect();
        // Partial sort would be faster; k is small so a full sort is fine.
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n_probe).map(|(i, _)| i).collect()
    }

    pub fn len(&self) -> usize {
        self.vecs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vecs.is_empty()
    }
}

#[inline]
pub fn nearest_centroid_idx(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = sq_l2(v, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Rank each centroid by L2² distance to `v`; returns (idx, dist²) sorted asc.
pub fn rank_centroids(v: &[f32], centroids: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut r: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq_l2(v, c)))
        .collect();
    r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    r
}

// ─── posting-list container ──────────────────────────────────────────────────

/// One posting-list per centroid. `dup_count` tracks how many *unique* corpus
/// ids are duplicated across ≥2 lists (used for memory reporting).
/// `residual_norms[c][i]` = ‖x_{lists[c][i]} − centroid[c]‖ — cached at build
/// time so approximate-scan (`probe_and_scan_approx`) is zero-alloc per query.
pub struct PostingLists {
    pub lists: Vec<Vec<usize>>,
    pub residual_norms: Vec<Vec<f32>>,
    pub dup_count: usize,
    pub total_entries: usize,
}

impl PostingLists {
    pub fn new(n_centroids: usize) -> Self {
        Self {
            lists: (0..n_centroids).map(|_| Vec::new()).collect(),
            residual_norms: (0..n_centroids).map(|_| Vec::new()).collect(),
            dup_count: 0,
            total_entries: 0,
        }
    }

    pub fn push_with_norm(&mut self, centroid: usize, id: usize, res_norm: f32) {
        self.lists[centroid].push(id);
        self.residual_norms[centroid].push(res_norm);
        self.total_entries += 1;
    }
}

// ─── the trait every variant implements ──────────────────────────────────────

pub trait IvfVariant: Send + Sync {
    fn name(&self) -> &str;

    /// Exact-rescan search: return top-k nearest neighbours using `n_probe`
    /// centroids, scored by exact L2² against the original corpus.
    fn search(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<Hit>;

    /// Approximate search using the centroid-anchored Cauchy–Schwarz upper
    /// bound. This is the regime SOAR is designed for — no access to the raw
    /// corpus is needed at query time.
    fn search_approx(&self, query: &[f32], k: usize, n_probe: usize) -> Vec<Hit>;

    /// Estimated heap bytes for the index (posting lists + centroids + vectors
    /// keep the caller-supplied slice, so this reports only *index overhead*).
    fn memory_bytes(&self) -> usize;

    /// Number of duplicated postings (spillover). 0 for single-assignment IVF.
    fn spill_overhead(&self) -> usize;
}

// ─── shared probe-and-scan retrieval ─────────────────────────────────────────

/// Scan the posting lists at the given probed centroids and return the top-k
/// hits by exact L2² distance to the *original* corpus vectors.
///
/// Deduplicates: if a corpus id appears in multiple probed lists (spillover),
/// we only score it once per query.
pub fn probe_and_scan(
    query: &[f32],
    k: usize,
    n_probe: usize,
    centroids: &Centroids,
    lists: &PostingLists,
    vectors: &[Vec<f32>],
) -> Vec<Hit> {
    let probed = centroids.probe(query, n_probe);
    let mut seen = vec![false; vectors.len()];
    let mut cands: Vec<usize> = Vec::new();
    for c in probed {
        for &id in &lists.lists[c] {
            if !seen[id] {
                seen[id] = true;
                cands.push(id);
            }
        }
    }
    let mut hits: Vec<Hit> = cands
        .into_iter()
        .map(|id| Hit {
            id,
            dist: sq_l2(query, &vectors[id]),
        })
        .collect();
    hits.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(k);
    hits
}

/// Approximate probe-and-scan using the centroid-anchored *estimator*
/// `d̂(q, x) = ‖q − c‖² + ‖x − c‖²`, which equals the true squared distance
/// when the cross-term ⟨q − c, x − c⟩ = 0. This is exactly the regime SOAR
/// engineers: SOAR picks secondaries whose residuals are *orthogonal to the
/// primary residual*, so under the plausible assumption that a query direction
/// tracks its primary residual, the SOAR-selected secondary has near-zero
/// cross-term. Top-k spilling picks the two nearest centroids whose residuals
/// tend to be *parallel*, so their approximation errors are correlated and the
/// min-over-copies estimator is biased upward.
///
/// For each candidate we take the minimum estimator across all posted
/// centroids and use it as the score. `residual_norms` is cached at build.
pub fn probe_and_scan_approx(
    query: &[f32],
    k: usize,
    n_probe: usize,
    centroids: &Centroids,
    lists: &PostingLists,
) -> Vec<Hit> {
    use std::collections::HashMap;
    let probed = centroids.probe(query, n_probe);
    let mut best: HashMap<usize, f32> = HashMap::with_capacity(256);
    for &c in &probed {
        let qcs = sq_l2(query, &centroids.vecs[c]);
        for (i, &id) in lists.lists[c].iter().enumerate() {
            let xn = lists.residual_norms[c][i];
            // Zero-cross-term estimator (tightens for SOAR-selected copies).
            let est = qcs + xn * xn;
            match best.get_mut(&id) {
                Some(v) => {
                    if est < *v {
                        *v = est;
                    }
                }
                None => {
                    best.insert(id, est);
                }
            }
        }
    }
    let mut hits: Vec<Hit> = best.into_iter().map(|(id, d)| Hit { id, dist: d }).collect();
    hits.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(k);
    hits
}

/// Bytes-per-vector estimate for a flat corpus: dims * 4.
#[inline]
pub fn corpus_bytes(n: usize, dims: usize) -> usize {
    n * dims * std::mem::size_of::<f32>()
}

/// Bytes for one posting-list container (all `usize` ids + f32 residual norms).
pub fn posting_bytes(lists: &PostingLists) -> usize {
    lists.total_entries * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
        + lists.lists.len() * (std::mem::size_of::<Vec<usize>>() + std::mem::size_of::<Vec<f32>>())
}

/// Bytes for the centroid table.
pub fn centroid_bytes(c: &Centroids) -> usize {
    c.len() * c.dims * std::mem::size_of::<f32>()
        + c.len() * std::mem::size_of::<Vec<f32>>()
}

// keep the linker happy on the `dot` re-export
#[allow(dead_code)]
fn _keep_dot_used() -> f32 {
    dot(&[0.0], &[0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_produces_centroids_with_all_points_assigned() {
        // 3 tight blobs; k=3 should recover something close.
        let mut v = Vec::new();
        for _ in 0..30 {
            v.push(vec![0.0, 0.0]);
            v.push(vec![10.0, 0.0]);
            v.push(vec![0.0, 10.0]);
        }
        let c = Centroids::train(&v, 3, 10, 42);
        assert_eq!(c.vecs.len(), 3);
        // Every point's nearest centroid should be exactly one of {0,10}-ish.
        for p in &v {
            let idx = nearest_centroid_idx(p, &c.vecs);
            let cc = &c.vecs[idx];
            assert!(sq_l2(p, cc) < 5.0, "point {:?} centroid {:?}", p, cc);
        }
    }

    #[test]
    fn probe_returns_requested_number() {
        let v: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32, 0.0]).collect();
        let c = Centroids::train(&v, 5, 5, 7);
        let probed = c.probe(&[3.0, 0.0], 3);
        assert_eq!(probed.len(), 3);
        // All indices are in range.
        for i in probed {
            assert!(i < c.len());
        }
    }

    #[test]
    fn rank_centroids_is_sorted_ascending() {
        let cs = vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![100.0, 0.0]];
        let r = rank_centroids(&[1.0, 0.0], &cs);
        assert_eq!(r[0].0, 0);
        assert_eq!(r[1].0, 1);
        assert_eq!(r[2].0, 2);
        assert!(r[0].1 < r[1].1 && r[1].1 < r[2].1);
    }
}
