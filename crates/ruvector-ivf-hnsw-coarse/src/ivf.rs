//! IVF index that delegates coarse-list selection to any [`IvfCoarse`]
//! implementation. Postings scan is identical across coarse quantizers so
//! recall / latency deltas are purely from the coarse quantizer.

use crate::{sq_l2, CoarseError, IvfCoarse};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration for [`Ivf`] construction.
#[derive(Debug, Clone, Copy)]
pub struct IvfConfig {
    /// Number of centroids `C`.
    pub c: usize,
    /// K-means iterations (Lloyd).
    pub kmeans_iters: usize,
    /// Seed for k-means initialization.
    pub seed: u64,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            c: 256,
            kmeans_iters: 15,
            seed: 0x1FF00_C0DE,
        }
    }
}

/// Timing / recall statistics from an IVF search sweep.
#[derive(Debug, Default, Clone)]
pub struct IvfStats {
    /// Total nanoseconds spent in `top_lists` calls.
    pub coarse_ns: u128,
    /// Total nanoseconds spent scanning posting lists.
    pub scan_ns: u128,
    /// Number of queries.
    pub queries: usize,
    /// Recall@k against exact ground truth.
    pub recall: f64,
}

impl IvfStats {
    /// Mean coarse-select microseconds per query.
    pub fn coarse_us(&self) -> f64 {
        self.coarse_ns as f64 / 1_000.0 / self.queries.max(1) as f64
    }
    /// Mean scan microseconds per query.
    pub fn scan_us(&self) -> f64 {
        self.scan_ns as f64 / 1_000.0 / self.queries.max(1) as f64
    }
    /// Total mean microseconds per query.
    pub fn total_us(&self) -> f64 {
        self.coarse_us() + self.scan_us()
    }
}

/// IVF index (flat, non-quantized postings) parameterized by coarse quantizer.
pub struct Ivf<C: IvfCoarse> {
    coarse: C,
    /// Row-major `C x d` centroids.
    pub centroids: Vec<f32>,
    /// Row-major `N x d` vectors (kept flat, no residual coding here).
    pub data: Vec<f32>,
    /// posting lists: postings[list_id] = Vec<row_id>.
    pub postings: Vec<Vec<u32>>,
    d: usize,
    n: usize,
}

impl<C: IvfCoarse> Ivf<C> {
    /// Train centroids, wire postings, and fit the coarse quantizer.
    pub fn build(
        data: &[f32],
        d: usize,
        cfg: IvfConfig,
        mut coarse: C,
    ) -> Result<Self, CoarseError> {
        if d == 0 || data.len() % d != 0 {
            return Err(CoarseError::DimensionMismatch {
                expected: d,
                got: data.len(),
            });
        }
        let n = data.len() / d;
        let centroids = kmeans(data, d, cfg.c, cfg.kmeans_iters, cfg.seed);
        coarse.fit(&centroids, d)?;
        // Assign every row to its nearest centroid via a flat scan --
        // done once at build, not on the hot path.
        let mut postings: Vec<Vec<u32>> = vec![Vec::new(); cfg.c];
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let mut best = 0u32;
            let mut best_d = f32::INFINITY;
            for c in 0..cfg.c {
                let dd = sq_l2(row, &centroids[c * d..(c + 1) * d]);
                if dd < best_d {
                    best_d = dd;
                    best = c as u32;
                }
            }
            postings[best as usize].push(i as u32);
        }
        Ok(Self {
            coarse,
            centroids,
            data: data.to_vec(),
            postings,
            d,
            n,
        })
    }

    /// Return top-k row IDs (ascending distance).
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<u32> {
        assert_eq!(query.len(), self.d);
        let lists = self.coarse.top_lists(query, nprobe);
        let mut out: Vec<(f32, u32)> = Vec::with_capacity(k * 4);
        for l in lists {
            for &row in &self.postings[l as usize] {
                let v = &self.data[row as usize * self.d..(row as usize + 1) * self.d];
                out.push((sq_l2(query, v), row));
            }
        }
        if out.is_empty() {
            return Vec::new();
        }
        let k = k.min(out.len());
        out.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        out.truncate(k);
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.into_iter().map(|(_, i)| i).collect()
    }

    /// Run an IVF search sweep with separated coarse / scan timing.
    pub fn timed_sweep(
        &self,
        queries: &[f32],
        k: usize,
        nprobe: usize,
        gt: &[Vec<u32>],
    ) -> IvfStats {
        let nq = queries.len() / self.d;
        assert_eq!(gt.len(), nq);
        let mut stats = IvfStats::default();
        stats.queries = nq;
        let mut hits: usize = 0;
        let mut denom: usize = 0;
        for qi in 0..nq {
            let q = &queries[qi * self.d..(qi + 1) * self.d];
            let t0 = std::time::Instant::now();
            let lists = self.coarse.top_lists(q, nprobe);
            let t1 = std::time::Instant::now();
            let mut out: Vec<(f32, u32)> = Vec::with_capacity(k * 4);
            for l in lists {
                for &row in &self.postings[l as usize] {
                    let v = &self.data[row as usize * self.d..(row as usize + 1) * self.d];
                    out.push((sq_l2(q, v), row));
                }
            }
            let take_k = k.min(out.len()).max(1);
            if !out.is_empty() {
                out.select_nth_unstable_by(take_k - 1, |a, b| {
                    a.0.partial_cmp(&b.0).unwrap()
                });
                out.truncate(take_k);
                out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            }
            let t2 = std::time::Instant::now();
            stats.coarse_ns += (t1 - t0).as_nanos();
            stats.scan_ns += (t2 - t1).as_nanos();
            let found: std::collections::HashSet<u32> = out.iter().map(|x| x.1).collect();
            for &g in &gt[qi] {
                if found.contains(&g) {
                    hits += 1;
                }
                denom += 1;
            }
        }
        stats.recall = hits as f64 / denom as f64;
        stats
    }

    /// N (row count).
    pub fn n(&self) -> usize {
        self.n
    }
    /// C (centroid count).
    pub fn c(&self) -> usize {
        self.centroids.len() / self.d
    }
    /// d (dim).
    pub fn d(&self) -> usize {
        self.d
    }
    /// Bytes used by the coarse quantizer beyond the centroid table itself.
    pub fn coarse_extra_bytes(&self) -> usize {
        self.coarse.extra_bytes()
    }
    /// Coarse quantizer name.
    pub fn coarse_name(&self) -> &'static str {
        self.coarse.name()
    }
}

/// Simple deterministic Lloyd's k-means. Returns row-major `k x d` centroids.
pub fn kmeans(data: &[f32], d: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = data.len() / d;
    assert!(n >= k);
    let mut rng = StdRng::seed_from_u64(seed);
    // Init: pick `k` distinct random rows.
    let mut picked = std::collections::HashSet::new();
    let mut centroids = Vec::with_capacity(k * d);
    while picked.len() < k {
        let i: usize = rng.gen_range(0..n);
        if picked.insert(i) {
            centroids.extend_from_slice(&data[i * d..(i + 1) * d]);
        }
    }

    let mut assign = vec![0u32; n];
    for _ in 0..iters {
        // Assign.
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            let mut best = 0u32;
            let mut bd = f32::INFINITY;
            for c in 0..k {
                let dd = sq_l2(row, &centroids[c * d..(c + 1) * d]);
                if dd < bd {
                    bd = dd;
                    best = c as u32;
                }
            }
            assign[i] = best;
        }
        // Update.
        let mut sums = vec![0.0f32; k * d];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let a = assign[i] as usize;
            counts[a] += 1;
            for j in 0..d {
                sums[a * d + j] += data[i * d + j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Reseed empty cluster to a random row.
                let i: usize = rng.gen_range(0..n);
                centroids[c * d..(c + 1) * d].copy_from_slice(&data[i * d..(i + 1) * d]);
                continue;
            }
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                centroids[c * d + j] = sums[c * d + j] * inv;
            }
        }
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlatCoarse, HnswCoarse};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn gen(n: usize, d: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    fn ground_truth(data: &[f32], queries: &[f32], d: usize, k: usize) -> Vec<Vec<u32>> {
        let n = data.len() / d;
        let nq = queries.len() / d;
        let mut gt = Vec::with_capacity(nq);
        for qi in 0..nq {
            let q = &queries[qi * d..(qi + 1) * d];
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| (sq_l2(q, &data[i * d..(i + 1) * d]), i as u32))
                .collect();
            scored.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            scored.truncate(k);
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            gt.push(scored.into_iter().map(|(_, i)| i).collect());
        }
        gt
    }

    #[test]
    fn ivf_hnsw_matches_ivf_flat_recall() {
        let d = 16;
        let n = 2000;
        let data = gen(n, d, 10);
        let queries = gen(50, d, 11);
        let gt = ground_truth(&data, &queries, d, 10);

        let cfg = IvfConfig {
            c: 64,
            kmeans_iters: 8,
            seed: 42,
        };
        let ivf_flat = Ivf::build(&data, d, cfg, FlatCoarse::new()).unwrap();
        let ivf_hnsw = Ivf::build(
            &data,
            d,
            cfg,
            HnswCoarse::with_config(super::super::hnsw::HnswConfig {
                ef_construction: 100,
                ef_search: 32,
                ..Default::default()
            }),
        )
        .unwrap();

        let sf = ivf_flat.timed_sweep(&queries, 10, 8, &gt);
        let sh = ivf_hnsw.timed_sweep(&queries, 10, 8, &gt);
        // Match within 3 percentage points of recall.
        assert!(
            (sh.recall - sf.recall).abs() < 0.03,
            "recall drift: flat={:.3} hnsw={:.3}",
            sf.recall,
            sh.recall
        );
        assert!(sf.recall > 0.5, "sanity: ivf-flat recall too low");
    }
}
