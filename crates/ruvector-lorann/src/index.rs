//! Index backends — `BruteForceIndex`, `IvfIndex`, `LoRannIndex`.
//!
//! All three implement the `InnerProductIndex` trait so the benchmark binary
//! and downstream code can A/B them without conditionals. Vectors are stored
//! L2-normalized so inner-product top-k coincides with cosine top-k.

use crate::{
    dot, l2_normalize,
    kmeans::KMeans,
    lowrank::{gram, matmul_xv, project_q, top_eigvecs},
};

/// A scored neighbor result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// Vector id (insertion order).
    pub id: u32,
    /// Inner-product score against the query (cosine sim for L2-normalized data).
    pub score: f32,
}

/// Swappable backend trait.
pub trait InnerProductIndex {
    /// Train/build the index from row-major `[n × d]` data. Implementations are
    /// allowed to mutate (e.g., L2-normalize) the buffer they retain internally.
    fn train(&mut self, x: Vec<f32>, n: usize, d: usize);
    /// Search for the top-`k` neighbors by inner product. The query may be any
    /// magnitude — implementations L2-normalize internally to match storage.
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor>;
    /// Identifier for logging/benchmarks.
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// BruteForceIndex — exact O(nd) baseline.
// ---------------------------------------------------------------------------

/// Exact brute-force inner-product top-k.
#[derive(Default)]
pub struct BruteForceIndex {
    x: Vec<f32>,
    n: usize,
    d: usize,
}

impl InnerProductIndex for BruteForceIndex {
    fn train(&mut self, mut x: Vec<f32>, n: usize, d: usize) {
        assert_eq!(x.len(), n * d);
        for i in 0..n {
            let mut row = &mut x[i * d..(i + 1) * d];
            l2_normalize(&mut row);
        }
        self.x = x;
        self.n = n;
        self.d = d;
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        assert_eq!(q.len(), self.d);
        let mut qn = q.to_vec();
        l2_normalize(&mut qn);
        let mut heap = TopK::new(k);
        for i in 0..self.n {
            let s = dot(&qn, &self.x[i * self.d..(i + 1) * self.d]);
            heap.push(Neighbor { id: i as u32, score: s });
        }
        heap.into_sorted()
    }
    fn name(&self) -> &'static str {
        "brute"
    }
}

// ---------------------------------------------------------------------------
// IvfIndex — k-means + nprobe exact scoring inside selected clusters.
// ---------------------------------------------------------------------------

/// IVF configuration.
#[derive(Debug, Clone, Copy)]
pub struct IvfConfig {
    /// Number of clusters.
    pub n_clusters: usize,
    /// `nprobe` — clusters to scan per query.
    pub nprobe: usize,
    /// k-means iterations.
    pub kmeans_iters: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for IvfConfig {
    fn default() -> Self {
        IvfConfig { n_clusters: 16, nprobe: 4, kmeans_iters: 25, seed: 1 }
    }
}

/// IVF backend — exact inner-product inside the top-`nprobe` clusters.
pub struct IvfIndex {
    cfg: IvfConfig,
    d: usize,
    km: Option<KMeans>,
    /// Per-cluster row-major data `[n_c × d]` and the original ids.
    clusters: Vec<(Vec<f32>, Vec<u32>)>,
}

impl IvfIndex {
    /// New index from config.
    pub fn new(cfg: IvfConfig) -> Self {
        IvfIndex { cfg, d: 0, km: None, clusters: Vec::new() }
    }
}

impl InnerProductIndex for IvfIndex {
    fn train(&mut self, mut x: Vec<f32>, n: usize, d: usize) {
        assert_eq!(x.len(), n * d);
        for i in 0..n {
            let mut row = &mut x[i * d..(i + 1) * d];
            l2_normalize(&mut row);
        }
        let km = KMeans::train(&x, n, d, self.cfg.n_clusters, self.cfg.kmeans_iters, self.cfg.seed);
        let mut buckets: Vec<(Vec<f32>, Vec<u32>)> =
            (0..self.cfg.n_clusters).map(|_| (Vec::new(), Vec::new())).collect();
        for i in 0..n {
            let c = km.assignments[i];
            buckets[c].0.extend_from_slice(&x[i * d..(i + 1) * d]);
            buckets[c].1.push(i as u32);
        }
        self.d = d;
        self.km = Some(km);
        self.clusters = buckets;
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        let km = self.km.as_ref().expect("IvfIndex: not trained");
        assert_eq!(q.len(), self.d);
        let mut qn = q.to_vec();
        l2_normalize(&mut qn);
        let selected = pick_clusters(&qn, &km.centroids, self.cfg.n_clusters, self.d, self.cfg.nprobe);
        let mut heap = TopK::new(k);
        for c in selected {
            let (rows, ids) = &self.clusters[c];
            let n_c = ids.len();
            for j in 0..n_c {
                let s = dot(&qn, &rows[j * self.d..(j + 1) * self.d]);
                heap.push(Neighbor { id: ids[j], score: s });
            }
        }
        heap.into_sorted()
    }
    fn name(&self) -> &'static str {
        "ivf"
    }
}

// ---------------------------------------------------------------------------
// LoRannIndex — IVF + per-cluster reduced-rank approximation + exact rerank.
// ---------------------------------------------------------------------------

/// LoRANN configuration.
#[derive(Debug, Clone, Copy)]
pub struct LoRannConfig {
    /// IVF clustering parameters.
    pub ivf: IvfConfig,
    /// Reduced rank `r`.
    pub rank: usize,
    /// Subspace-iteration steps per cluster.
    pub subspace_iters: usize,
    /// Candidates kept per cluster for exact rerank (in addition to k).
    pub rerank_per_cluster: usize,
}

impl Default for LoRannConfig {
    fn default() -> Self {
        LoRannConfig {
            ivf: IvfConfig::default(),
            rank: 8,
            subspace_iters: 20,
            rerank_per_cluster: 32,
        }
    }
}

/// LoRANN backend.
pub struct LoRannIndex {
    cfg: LoRannConfig,
    d: usize,
    km: Option<KMeans>,
    /// Per-cluster: original rows `[n_c × d]`, ids, reduced coords `A_c` `[n_c × r]`,
    /// basis `V_c` `[d × r]`.
    clusters: Vec<ClusterBlock>,
}

struct ClusterBlock {
    rows: Vec<f32>,
    ids: Vec<u32>,
    a: Vec<f32>,
    v: Vec<f32>,
    n_c: usize,
}

impl LoRannIndex {
    /// New index from config.
    pub fn new(cfg: LoRannConfig) -> Self {
        LoRannIndex { cfg, d: 0, km: None, clusters: Vec::new() }
    }
}

impl InnerProductIndex for LoRannIndex {
    fn train(&mut self, mut x: Vec<f32>, n: usize, d: usize) {
        assert_eq!(x.len(), n * d);
        let r = self.cfg.rank.min(d);
        for i in 0..n {
            let mut row = &mut x[i * d..(i + 1) * d];
            l2_normalize(&mut row);
        }
        let km =
            KMeans::train(&x, n, d, self.cfg.ivf.n_clusters, self.cfg.ivf.kmeans_iters, self.cfg.ivf.seed);

        // Bucket by cluster.
        let mut buckets: Vec<(Vec<f32>, Vec<u32>)> =
            (0..self.cfg.ivf.n_clusters).map(|_| (Vec::new(), Vec::new())).collect();
        for i in 0..n {
            let c = km.assignments[i];
            buckets[c].0.extend_from_slice(&x[i * d..(i + 1) * d]);
            buckets[c].1.push(i as u32);
        }

        // Per-cluster reduced-rank fit.
        let mut clusters = Vec::with_capacity(self.cfg.ivf.n_clusters);
        for (c, (rows, ids)) in buckets.into_iter().enumerate() {
            let n_c = ids.len();
            if n_c == 0 {
                clusters.push(ClusterBlock { rows, ids, a: Vec::new(), v: Vec::new(), n_c });
                continue;
            }
            let m = gram(&rows, n_c, d);
            let v = top_eigvecs(&m, d, r, self.cfg.subspace_iters, self.cfg.ivf.seed ^ (c as u64).wrapping_mul(0x9E37_79B9));
            let a = matmul_xv(&rows, n_c, d, &v, r);
            clusters.push(ClusterBlock { rows, ids, a, v, n_c });
        }

        self.d = d;
        self.km = Some(km);
        self.clusters = clusters;
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        let km = self.km.as_ref().expect("LoRannIndex: not trained");
        assert_eq!(q.len(), self.d);
        let mut qn = q.to_vec();
        l2_normalize(&mut qn);
        let selected = pick_clusters(&qn, &km.centroids, self.cfg.ivf.n_clusters, self.d, self.cfg.ivf.nprobe);
        let r = self.cfg.rank.min(self.d);
        let mut heap = TopK::new(k);

        for c in selected {
            let cb = &self.clusters[c];
            if cb.n_c == 0 {
                continue;
            }
            let qr = project_q(&qn, &cb.v, self.d, r);
            // Approximate scores in O(n_c · r).
            let mut approx = Vec::with_capacity(cb.n_c);
            for i in 0..cb.n_c {
                let mut s = 0.0_f32;
                for j in 0..r {
                    s += cb.a[i * r + j] * qr[j];
                }
                approx.push((i, s));
            }
            // Pick top `rerank_per_cluster` candidates by approximate score.
            let keep = self.cfg.rerank_per_cluster.min(cb.n_c);
            approx.select_nth_unstable_by(keep.saturating_sub(1).min(cb.n_c - 1), |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Exact rerank.
            for (i, _approx) in approx.iter().take(keep) {
                let exact = dot(&qn, &cb.rows[*i * self.d..(*i + 1) * self.d]);
                heap.push(Neighbor { id: cb.ids[*i], score: exact });
            }
        }
        heap.into_sorted()
    }
    fn name(&self) -> &'static str {
        "lorann"
    }
}

// ---------------------------------------------------------------------------
// Shared helpers.
// ---------------------------------------------------------------------------

fn pick_clusters(q: &[f32], centroids: &[f32], k: usize, d: usize, nprobe: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = (0..k)
        .map(|c| {
            let mut s = 0.0_f32;
            for j in 0..d {
                s += q[j] * centroids[c * d + j];
            }
            (c, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(nprobe.min(k)).map(|(c, _)| c).collect()
}

/// Bounded min-heap of top-`k` results by score.
struct TopK {
    cap: usize,
    heap: Vec<Neighbor>,
}

impl TopK {
    fn new(cap: usize) -> Self {
        TopK { cap, heap: Vec::with_capacity(cap.max(1)) }
    }
    fn push(&mut self, n: Neighbor) {
        if self.heap.len() < self.cap {
            self.heap.push(n);
            // Maintain min-heap by score.
            self.siftup();
        } else if !self.heap.is_empty() && n.score > self.heap[0].score {
            self.heap[0] = n;
            self.siftdown();
        }
    }
    fn siftup(&mut self) {
        let mut i = self.heap.len() - 1;
        while i > 0 {
            let p = (i - 1) / 2;
            if self.heap[i].score < self.heap[p].score {
                self.heap.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }
    fn siftdown(&mut self) {
        let n = self.heap.len();
        let mut i = 0;
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut best = i;
            if l < n && self.heap[l].score < self.heap[best].score {
                best = l;
            }
            if r < n && self.heap[r].score < self.heap[best].score {
                best = r;
            }
            if best == i {
                break;
            }
            self.heap.swap(i, best);
            i = best;
        }
    }
    fn into_sorted(mut self) -> Vec<Neighbor> {
        self.heap.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        self.heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_dataset(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = crate::Lcg::new(seed);
        let mut x = vec![0.0_f32; n * d];
        for k in 0..n * d {
            x[k] = rng.next_f32() - 0.5;
        }
        x
    }

    #[test]
    fn brute_finds_self_as_top1() {
        let n = 64;
        let d = 12;
        let x = synth_dataset(n, d, 1);
        let mut idx = BruteForceIndex::default();
        idx.train(x.clone(), n, d);
        for i in 0..8 {
            let q = &x[i * d..(i + 1) * d];
            let top = idx.search(q, 1);
            assert_eq!(top[0].id, i as u32, "self should be top1 for i={i}");
            assert!(top[0].score > 0.99, "self-similarity should be ~1 ({})", top[0].score);
        }
    }

    #[test]
    fn topk_heap_returns_sorted_top_k() {
        let mut h = TopK::new(3);
        for (i, s) in [(0, 0.1), (1, 0.9), (2, 0.5), (3, 0.7), (4, 0.2)] {
            h.push(Neighbor { id: i, score: s });
        }
        let out = h.into_sorted();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].id, 1);
        assert_eq!(out[1].id, 3);
        assert_eq!(out[2].id, 2);
    }
}
