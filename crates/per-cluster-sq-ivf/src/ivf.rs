//! IVF driver with pluggable storage backend.

use crate::kmeans::{kmeans, l2_sq, probe_clusters};
use crate::sq::{encode_vec, fit_ranges, l2_sq_decoded};

/// A storage backend owns the per-vector representation and residual-distance math.
pub trait IvfBackend: Send + Sync {
    fn train(&mut self, data: &[f32], d: usize, assignments: &[u32], centroids: &[f32], k: usize);
    /// Scored squared-L2 between `query_residual` (query - centroid of cluster `cid`)
    /// and the stored vector `local_idx` within cluster `cid`.
    fn score(&self, cid: u32, local_idx: usize, query_residual: &[f32]) -> f32;
    fn storage_bytes_per_vec(&self, d: usize) -> f64;
    fn name(&self) -> &'static str;
}

pub struct Float32Backend {
    // residuals stored contiguously per cluster
    pub clusters: Vec<Vec<f32>>, // clusters[cid] = flat residuals [n_c * d]
    pub d: usize,
}

impl Float32Backend {
    pub fn new() -> Self { Self { clusters: Vec::new(), d: 0 } }
}

impl IvfBackend for Float32Backend {
    fn train(&mut self, data: &[f32], d: usize, assignments: &[u32], centroids: &[f32], k: usize) {
        self.d = d;
        self.clusters = vec![Vec::new(); k];
        let n = data.len() / d;
        for i in 0..n {
            let c = assignments[i] as usize;
            let base = c * d;
            let vb = i * d;
            for j in 0..d {
                self.clusters[c].push(data[vb + j] - centroids[base + j]);
            }
        }
    }
    fn score(&self, cid: u32, local_idx: usize, query_residual: &[f32]) -> f32 {
        let d = self.d;
        let v = &self.clusters[cid as usize][local_idx * d..(local_idx + 1) * d];
        l2_sq(v, query_residual)
    }
    fn storage_bytes_per_vec(&self, d: usize) -> f64 { (d * 4) as f64 }
    fn name(&self) -> &'static str { "float32" }
}

pub struct GlobalSq4Backend {
    pub clusters: Vec<Vec<u8>>, // packed codes
    pub lo: Vec<f32>,            // d
    pub hi: Vec<f32>,            // d
    pub d: usize,
}

impl GlobalSq4Backend {
    pub fn new() -> Self { Self { clusters: Vec::new(), lo: Vec::new(), hi: Vec::new(), d: 0 } }
}

impl IvfBackend for GlobalSq4Backend {
    fn train(&mut self, data: &[f32], d: usize, assignments: &[u32], centroids: &[f32], k: usize) {
        self.d = d;
        self.clusters = vec![Vec::new(); k];
        let n = data.len() / d;
        // compute residuals globally first for range fit
        let mut residuals = vec![0.0f32; n * d];
        for i in 0..n {
            let c = assignments[i] as usize;
            let base = c * d;
            let vb = i * d;
            for j in 0..d {
                residuals[vb + j] = data[vb + j] - centroids[base + j];
            }
        }
        let (lo, hi) = fit_ranges(&residuals, d);
        self.lo = lo;
        self.hi = hi;
        let packed_len = (d + 1) / 2;
        let mut tmp = vec![0u8; packed_len];
        for i in 0..n {
            let c = assignments[i] as usize;
            encode_vec(&residuals[i * d..(i + 1) * d], &self.lo, &self.hi, &mut tmp);
            self.clusters[c].extend_from_slice(&tmp);
        }
    }
    fn score(&self, cid: u32, local_idx: usize, query_residual: &[f32]) -> f32 {
        let packed_len = (self.d + 1) / 2;
        let p = &self.clusters[cid as usize]
            [local_idx * packed_len..(local_idx + 1) * packed_len];
        l2_sq_decoded(p, &self.lo, &self.hi, query_residual)
    }
    fn storage_bytes_per_vec(&self, d: usize) -> f64 {
        ((d + 1) / 2) as f64
        // Global tables are amortized to ~0 per vector at scale.
    }
    fn name(&self) -> &'static str { "global-sq4" }
}

pub struct PerClusterSq4Backend {
    pub clusters: Vec<Vec<u8>>,
    pub cluster_lo: Vec<Vec<f32>>, // k x d
    pub cluster_hi: Vec<Vec<f32>>, // k x d
    pub d: usize,
    pub k: usize,
}

impl PerClusterSq4Backend {
    pub fn new() -> Self {
        Self { clusters: Vec::new(), cluster_lo: Vec::new(), cluster_hi: Vec::new(), d: 0, k: 0 }
    }
}

impl IvfBackend for PerClusterSq4Backend {
    fn train(&mut self, data: &[f32], d: usize, assignments: &[u32], centroids: &[f32], k: usize) {
        self.d = d;
        self.k = k;
        self.clusters = vec![Vec::new(); k];
        self.cluster_lo = vec![vec![0.0; d]; k];
        self.cluster_hi = vec![vec![0.0; d]; k];
        let n = data.len() / d;

        // Group residuals by cluster, fit ranges locally, encode.
        let mut per_cluster_residuals: Vec<Vec<f32>> = vec![Vec::new(); k];
        for i in 0..n {
            let c = assignments[i] as usize;
            let base = c * d;
            let vb = i * d;
            for j in 0..d {
                per_cluster_residuals[c].push(data[vb + j] - centroids[base + j]);
            }
        }
        let packed_len = (d + 1) / 2;
        let mut tmp = vec![0u8; packed_len];
        for c in 0..k {
            if per_cluster_residuals[c].is_empty() {
                // keep zero ranges, cluster never visited
                self.cluster_hi[c] = vec![1e-6; d];
                continue;
            }
            let (lo, hi) = fit_ranges(&per_cluster_residuals[c], d);
            self.cluster_lo[c] = lo;
            self.cluster_hi[c] = hi;
            let n_c = per_cluster_residuals[c].len() / d;
            for i in 0..n_c {
                encode_vec(
                    &per_cluster_residuals[c][i * d..(i + 1) * d],
                    &self.cluster_lo[c],
                    &self.cluster_hi[c],
                    &mut tmp,
                );
                self.clusters[c].extend_from_slice(&tmp);
            }
        }
    }
    fn score(&self, cid: u32, local_idx: usize, query_residual: &[f32]) -> f32 {
        let packed_len = (self.d + 1) / 2;
        let c = cid as usize;
        let p = &self.clusters[c]
            [local_idx * packed_len..(local_idx + 1) * packed_len];
        l2_sq_decoded(p, &self.cluster_lo[c], &self.cluster_hi[c], query_residual)
    }
    fn storage_bytes_per_vec(&self, d: usize) -> f64 {
        // Codes + amortized per-cluster tables (k * 2d * 4 bytes) / n.
        // Caller computes exact overhead with knowledge of n/k; we report code-only here.
        ((d + 1) / 2) as f64
    }
    fn name(&self) -> &'static str { "per-cluster-sq4" }
}

/// IVF index over any backend. Owns centroids + per-cluster id lists.
pub struct IvfIndex<B: IvfBackend> {
    pub centroids: Vec<f32>, // k * d
    pub d: usize,
    pub k: usize,
    pub cluster_ids: Vec<Vec<u32>>, // cluster_ids[cid] = original vector ids
    pub backend: B,
}

impl<B: IvfBackend> IvfIndex<B> {
    pub fn train(data: &[f32], n: usize, d: usize, k: usize, max_iter: usize, seed: u64, mut backend: B) -> Self {
        let km = kmeans(data, n, d, k, max_iter, seed);
        let mut cluster_ids: Vec<Vec<u32>> = vec![Vec::new(); k];
        for i in 0..n { cluster_ids[km.assignments[i] as usize].push(i as u32); }
        backend.train(data, d, &km.assignments, &km.centroids, k);
        Self { centroids: km.centroids, d, k, cluster_ids, backend }
    }

    /// Search top-k nearest with `nprobe` clusters visited.
    pub fn search(&self, query: &[f32], top: usize, nprobe: usize) -> Vec<(u32, f32)> {
        let probes = probe_clusters(&self.centroids, self.d, self.k, query, nprobe);
        let mut heap: Vec<(u32, f32)> = Vec::new();
        // precompute residual per probe cluster
        let mut q_res = vec![0.0f32; self.d];
        for &cid in &probes {
            let c = cid as usize;
            for j in 0..self.d {
                q_res[j] = query[j] - self.centroids[c * self.d + j];
            }
            let ids = &self.cluster_ids[c];
            for (local_idx, &gid) in ids.iter().enumerate() {
                let s = self.backend.score(cid, local_idx, &q_res);
                heap.push((gid, s));
            }
        }
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        heap.truncate(top);
        heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{make_gaussian_mixture, exact_topk};

    fn recall(gt: &[u32], got: &[u32]) -> f32 {
        let mut hits = 0;
        for g in got { if gt.contains(g) { hits += 1; } }
        hits as f32 / gt.len() as f32
    }

    #[test]
    fn ivf_float32_matches_exact_at_full_probe() {
        let (data, _) = make_gaussian_mixture(512, 32, 8, 42);
        let idx = IvfIndex::train(&data, 512, 32, 8, 15, 1, Float32Backend::new());
        let q = &data[0..32];
        let gt = exact_topk(&data, 32, q, 10);
        let got = idx.search(q, 10, 8);
        let gt_ids: Vec<u32> = gt.iter().map(|x| x.0).collect();
        let got_ids: Vec<u32> = got.iter().map(|x| x.0).collect();
        assert!(recall(&gt_ids, &got_ids) >= 0.99);
    }

    #[test]
    fn per_cluster_beats_global_at_fixed_bits() {
        let (data, _) = make_gaussian_mixture(2048, 32, 16, 7);
        let idx_g = IvfIndex::train(&data, 2048, 32, 16, 20, 1, GlobalSq4Backend::new());
        let idx_pc = IvfIndex::train(&data, 2048, 32, 16, 20, 1, PerClusterSq4Backend::new());
        // Encoding error: smaller is better.
        let mut err_g = 0.0f32;
        let mut err_pc = 0.0f32;
        for i in 0..256 {
            let v = &data[i * 32..(i + 1) * 32];
            // find cluster for this i
            let c_g = {
                let km = &idx_g;
                let mut best = 0usize; let mut bd = f32::INFINITY;
                for c in 0..km.k {
                    let dist = crate::kmeans::l2_sq(v, &km.centroids[c * 32..(c + 1) * 32]);
                    if dist < bd { bd = dist; best = c; }
                }
                best
            };
            // residual
            let mut res = vec![0.0f32; 32];
            for j in 0..32 { res[j] = v[j] - idx_g.centroids[c_g * 32 + j]; }
            // find local idx in cluster
            let local_g = idx_g.cluster_ids[c_g].iter().position(|&g| g as usize == i).unwrap();
            let local_pc = idx_pc.cluster_ids[c_g].iter().position(|&g| g as usize == i).unwrap();
            let s_g = idx_g.backend.score(c_g as u32, local_g, &res);
            let s_pc = idx_pc.backend.score(c_g as u32, local_pc, &res);
            err_g += s_g;
            err_pc += s_pc;
        }
        // per-cluster must have strictly lower reconstruction error on self-queries
        assert!(err_pc < err_g, "pc_err={} global_err={}", err_pc, err_g);
    }
}
