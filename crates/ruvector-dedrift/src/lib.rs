//! DEDRIFT — incremental IVF rebalancing under content drift.
//!
//! This crate contains:
//!   * a minimal float32 IVF (inverted file) index that can be insert-only or
//!     incrementally rebalanced,
//!   * three rebalancing policies — `Split`, `Lazy`, `Hybrid` — that follow the
//!     DEDRIFT paper (Baranchuk et al., ICCV 2023) but are implemented from
//!     scratch in pure Rust against synthetic data,
//!   * a deterministic Gaussian-mixture drift generator that lets the demo
//!     binary and benches measure recall@10 versus a freshly-rebuilt oracle.
//!
//! Design notes:
//!   * Vectors are `Vec<f32>`; distances are squared-L2 throughout. Squared-L2
//!     and L2 induce the same ranking, so reranking is a no-op.
//!   * Centroids are owned by the `Ivf` struct (`Vec<Vec<f32>>`) and lists are
//!     `Vec<Vec<u32>>` of vector ids. We keep raw vectors in a flat `Vec<f32>`
//!     of length `n_vectors * dim` so a single `id -> vector` lookup is just
//!     an index into a contiguous slab.
//!   * No `unsafe`, no SIMD intrinsics — the goal is a reference impl, not the
//!     fastest possible IVF. The benchmarks measure *relative* cost of the
//!     three policies against a `FullRebuild` baseline.

pub mod dedrift;
pub mod drift_sim;

use std::time::Instant;

/// Plain IVF index: a flat list of centroids and per-centroid posting lists.
#[derive(Clone)]
pub struct Ivf {
    pub dim: usize,
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<Vec<u32>>,
    /// Flat contiguous storage of all inserted vectors, indexed by id.
    pub vectors: Vec<f32>,
    /// Cached count of inserted vectors (= vectors.len() / dim).
    pub n: u32,
}

impl Ivf {
    pub fn new(dim: usize, n_lists: usize) -> Self {
        Self {
            dim,
            centroids: Vec::with_capacity(n_lists),
            lists: vec![Vec::new(); n_lists],
            vectors: Vec::new(),
            n: 0,
        }
    }

    /// Train centroids via single-pass k-means on `train`.
    pub fn train(&mut self, train: &[Vec<f32>], n_iters: usize, seed: u64) {
        assert!(!train.is_empty(), "train set must be non-empty");
        let n_lists = self.lists.len();
        let mut rng = SmallRng::new(seed);
        let mut centroids: Vec<Vec<f32>> = (0..n_lists)
            .map(|_| train[rng.usize(train.len())].clone())
            .collect();

        let mut assign = vec![0usize; train.len()];
        for _ in 0..n_iters {
            for (i, v) in train.iter().enumerate() {
                assign[i] = nearest_centroid(v, &centroids);
            }
            let mut new = vec![vec![0.0f32; self.dim]; n_lists];
            let mut counts = vec![0u32; n_lists];
            for (i, v) in train.iter().enumerate() {
                let c = assign[i];
                counts[c] += 1;
                for d in 0..self.dim {
                    new[c][d] += v[d];
                }
            }
            for c in 0..n_lists {
                if counts[c] > 0 {
                    let inv = 1.0 / counts[c] as f32;
                    for d in 0..self.dim {
                        new[c][d] *= inv;
                    }
                    centroids[c] = std::mem::take(&mut new[c]);
                }
                // empty list: keep prior centroid (rare for our test sizes)
            }
        }
        self.centroids = centroids;
    }

    /// Append a vector; returns its global id.
    pub fn add(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.n;
        self.vectors.extend_from_slice(v);
        let c = nearest_centroid(v, &self.centroids);
        self.lists[c].push(id);
        self.n += 1;
        id
    }

    /// Search returning top-k ids sorted by ascending squared-L2.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<u32> {
        assert_eq!(query.len(), self.dim);
        // Rank centroids by distance to query.
        let mut probes: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq_l2(query, c)))
            .collect();
        let np = nprobe.min(probes.len());
        if np > 0 && np < probes.len() {
            probes.select_nth_unstable_by(np - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        probes.truncate(np);

        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for (cidx, _) in probes {
            for &id in &self.lists[cidx] {
                let v = self.vector(id);
                let d = sq_l2(query, v);
                if heap.len() < k {
                    heap.push((d, id));
                    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                } else if d < heap[0].0 {
                    heap[0] = (d, id);
                    heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }
            }
        }
        heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        heap.into_iter().map(|(_, id)| id).collect()
    }

    /// Brute-force top-k against all inserted vectors. Used by the harness to
    /// compute ground truth (the oracle).
    pub fn brute(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for id in 0..self.n {
            let v = self.vector(id);
            let d = sq_l2(query, v);
            if heap.len() < k {
                heap.push((d, id));
                heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            } else if d < heap[0].0 {
                heap[0] = (d, id);
                heap.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            }
        }
        heap.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        heap.into_iter().map(|(_, id)| id).collect()
    }

    pub fn vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    pub fn n_lists(&self) -> usize {
        self.lists.len()
    }

    /// Approximate bytes for centroids + list ids + raw vectors.
    pub fn bytes(&self) -> usize {
        let centroids = self.centroids.iter().map(|c| c.len() * 4).sum::<usize>();
        let lists = self.lists.iter().map(|l| l.len() * 4).sum::<usize>();
        let vecs = self.vectors.len() * 4;
        centroids + lists + vecs
    }

    /// Pure-data drift signal: sum over lists of the squared distance from
    /// each member to its (current) centroid. Used as a cheap proxy for
    /// "how stale are the centroids" by the Lazy policy.
    pub fn drift_score(&self) -> f32 {
        let mut acc = 0.0f32;
        for (c, list) in self.centroids.iter().zip(self.lists.iter()) {
            for &id in list {
                acc += sq_l2(c, self.vector(id));
            }
        }
        acc
    }
}

/// Squared L2 distance, scalar fallback. Cheap and branchless enough that the
/// autovectorizer turns it into AVX2 on x86-64.
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

#[inline]
pub fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>]) -> usize {
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

/// Compute recall@k between `pred` and `truth`.
pub fn recall_at_k(pred: &[u32], truth: &[u32]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let mut hit = 0u32;
    for t in truth {
        if pred.iter().any(|p| p == t) {
            hit += 1;
        }
    }
    hit as f32 / truth.len() as f32
}

/// Tiny deterministic RNG (xorshift64*) so benchmarks don't drag in the full
/// `rand` runtime at hot paths.
pub struct SmallRng(u64);
impl SmallRng {
    pub fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa fill -> [0,1)
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
    }
    #[inline]
    pub fn usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
    /// Box-Muller standard normal.
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.next_f32() + 1e-9).min(1.0);
        let u2 = self.next_f32();
        ((-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()) as f32
    }
}

/// Convenience: time a closure in milliseconds.
pub fn time_ms<R>(mut f: impl FnMut() -> R) -> (R, f64) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed().as_secs_f64() * 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_l2_basic() {
        assert!((sq_l2(&[0., 0., 0.], &[1., 2., 2.]) - 9.0).abs() < 1e-6);
    }

    #[test]
    fn ivf_recall_no_drift_is_high() {
        // 4-d, 200 points, 8 lists, nprobe=4 => recall@1 should be >0.9 vs brute.
        let dim = 4;
        let mut rng = SmallRng::new(42);
        let train: Vec<Vec<f32>> = (0..200)
            .map(|_| (0..dim).map(|_| rng.normal()).collect())
            .collect();
        let mut ivf = Ivf::new(dim, 8);
        ivf.train(&train, 8, 7);
        for v in &train {
            ivf.add(v);
        }
        let mut hit = 0;
        let mut total = 0;
        for q in train.iter().take(50) {
            let p = ivf.search(q, 1, 4);
            let t = ivf.brute(q, 1);
            if p == t {
                hit += 1;
            }
            total += 1;
        }
        let r = hit as f32 / total as f32;
        assert!(r > 0.9, "recall {} too low", r);
    }
}
