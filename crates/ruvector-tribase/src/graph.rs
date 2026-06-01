//! Tiny in-memory flat graph index built by approximate kNN over a Gaussian
//! point cloud. Not a serious graph index — this is the substrate the three
//! search variants ride on, so we measure pruning effectiveness rather than
//! absolute QPS.

use rand::SeedableRng;
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use crate::metric::l2_sq;

pub struct FlatGraph {
    dim: usize,
    n: usize,
    /// Flat vectors: n * dim contiguous f32.
    data: Vec<f32>,
    /// Adjacency list: n * out_degree neighbors.
    adj: Vec<u32>,
    out_degree: usize,
}

impl FlatGraph {
    pub fn dim(&self) -> usize { self.dim }
    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn out_degree(&self) -> usize { self.out_degree }

    #[inline]
    pub fn vector(&self, id: u32) -> &[f32] {
        let off = id as usize * self.dim;
        &self.data[off..off + self.dim]
    }

    #[inline]
    pub fn neighbors(&self, id: u32) -> &[u32] {
        let off = id as usize * self.out_degree;
        &self.adj[off..off + self.out_degree]
    }

    /// Generate `n` Gaussian-clustered points in `d` dims and connect each to
    /// `m` nearest neighbors. Deterministic given `seed`.
    pub fn random_knn(n: usize, d: usize, m: usize, seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // 8 cluster centers — gives the triangle inequality something to bite on.
        let n_clusters = 8.min(n);
        let mut centers: Vec<f32> = Vec::with_capacity(n_clusters * d);
        for _ in 0..n_clusters * d {
            centers.push(rng.gen_range(-4.0..4.0));
        }

        let mut data = Vec::with_capacity(n * d);
        for i in 0..n {
            let c = i % n_clusters;
            for j in 0..d {
                let v = centers[c * d + j] + rng.gen_range(-0.5..0.5);
                data.push(v);
            }
        }

        // Brute-force kNN graph build — fine for PoC sizes (n <= a few thousand).
        let mut adj = vec![0u32; n * m];
        for i in 0..n {
            let v = &data[i * d..(i + 1) * d];
            let mut dists: Vec<(u32, f32)> = (0..n as u32)
                .filter(|&j| j as usize != i)
                .map(|j| {
                    let w = &data[j as usize * d..(j as usize + 1) * d];
                    (j, l2_sq(v, w))
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(m);
            for k in 0..m {
                adj[i * m + k] = if k < dists.len() { dists[k].0 } else { 0 };
            }
            // Reserve last 2 slots for random long-range edges to keep the graph connected
            // across clusters (otherwise beam search starting at node 0 can't reach other clusters).
            let long_range = 2.min(m);
            for r in 0..long_range {
                let rid = rng.gen_range(0..n) as u32;
                if rid as usize != i {
                    adj[i * m + (m - 1 - r)] = rid;
                }
            }
        }

        FlatGraph { dim: d, n, data, adj, out_degree: m }
    }
}
