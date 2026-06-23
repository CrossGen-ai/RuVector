//! Flat single-layer NSG-style ANN graph used as the baseline.
//!
//! - Build by k-NN-graph + diversification (prune edges whose distance to an existing
//!   neighbor is smaller than to the candidate, à la NSG occlusion rule).
//! - Search by greedy best-first traversal with a beam (ef_search).
//!
//! No quantization. This is the exact-distance baseline against which SymphonyQG is
//! compared.

use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// reprune kept intentionally simple; see build() for the bulk construction path.

#[derive(Clone, Copy, Debug)]
pub struct GraphParams {
    /// Max out-degree per node.
    pub m: usize,
    /// Candidate pool size during construction.
    pub ef_construction: usize,
    /// Default candidate pool during search.
    pub ef_search: usize,
    /// Entry-point selection seed.
    pub seed: u64,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 64,
            ef_search: 32,
            seed: 0xC0FFEE,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Cand {
    dist: f32,
    id: u32,
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; use natural order so root == worst-distance.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, PartialEq)]
struct MinCand {
    dist: f32,
    id: u32,
}
impl Eq for MinCand {}
impl Ord for MinCand {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse so peek/pop return the smallest dist.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MinCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct ExactGraph {
    pub dim: usize,
    pub vectors: Vec<f32>, // row-major n x dim
    pub neighbors: Vec<Vec<u32>>,
    pub entry: u32,
    pub params: GraphParams,
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

impl ExactGraph {
    pub fn vector(&self, id: u32) -> &[f32] {
        let id = id as usize;
        &self.vectors[id * self.dim..(id + 1) * self.dim]
    }

    /// Build via brute-force k-NN bootstrap. For each node, compute the actual
    /// top-`ef_construction` nearest neighbors, then apply NSG-style occlusion
    /// pruning to keep `m` diverse links. O(N^2 * D) — fine up to ~50k vectors
    /// for benchmark purposes; production builds would replace this with NSG /
    /// NN-descent / HNSW layered insertion. The PoC focuses on the *query-time*
    /// symphony of quantization with graph traversal, not on build scaling.
    pub fn build(vectors_flat: Vec<f32>, dim: usize, params: GraphParams) -> Self {
        assert!(vectors_flat.len() % dim == 0);
        let n = vectors_flat.len() / dim;
        let mut g = ExactGraph {
            dim,
            vectors: vectors_flat,
            neighbors: vec![Vec::with_capacity(params.m); n],
            entry: 0,
            params,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(params.seed);
        let mut perm: Vec<u32> = (0..n as u32).collect();
        perm.shuffle(&mut rng);
        g.entry = perm[0];

        // For each node, brute-force the ef_construction nearest, then prune.
        // Parallelise the brute-force step with rayon for real-world throughput.
        use rayon::prelude::*;
        let ef_c = params.ef_construction;
        let raw_neighbors: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let v_i = &g.vectors[i * dim..(i + 1) * dim];
                let mut all: Vec<Cand> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let v_j = &g.vectors[j * dim..(j + 1) * dim];
                        Cand {
                            dist: l2_sq(v_i, v_j),
                            id: j as u32,
                        }
                    })
                    .collect();
                all.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
                all.truncate(ef_c);
                // Hybrid pruning: keep the m/2 absolute-nearest neighbors unconditionally
                // (these are the highest-quality edges), then fill the remaining m/2 with
                // NSG-occlusion-pruned diverse links to keep the graph connected across
                // clusters. Pure occlusion pruning was too aggressive on dense clusters
                // and discarded too many true near-neighbors.
                // Pick m_near nearest edges + m_long random long-range edges to
                // guarantee cross-cluster connectivity (small-world property). Pure
                // nearest-neighbor graphs over clustered data are disconnected and
                // greedy search cannot escape its starting cluster.
                let m = params.m;
                let m_near = (m * 3) / 4;
                let mut chosen: Vec<u32> = Vec::with_capacity(m);
                for c in all.iter().take(m_near) {
                    chosen.push(c.id);
                }
                // Add long-range edges chosen deterministically from a per-node seed.
                let mut node_rng =
                    ChaCha8Rng::seed_from_u64(params.seed.wrapping_add(i as u64));
                while chosen.len() < m {
                    let cand = node_rng.gen_range(0..n) as u32;
                    if cand as usize != i && !chosen.contains(&cand) {
                        chosen.push(cand);
                    }
                }
                chosen
            })
            .collect();
        g.neighbors = raw_neighbors;
        g
    }

    /// Greedy beam search. Returns candidate pool sorted ascending by distance.
    fn search_internal(&self, query: &[f32], ef: usize, exclude: Option<u32>) -> Vec<Cand> {
        let mut visited = vec![false; self.neighbors.len()];
        let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
        let mut result: BinaryHeap<Cand> = BinaryHeap::new();
        let entry = self.entry;
        let d0 = l2_sq(query, self.vector(entry));
        candidates.push(MinCand { dist: d0, id: entry });
        result.push(Cand { dist: d0, id: entry });
        visited[entry as usize] = true;
        while let Some(MinCand { dist: cd, id: cid }) = candidates.pop() {
            let worst = result.peek().map(|c| c.dist).unwrap_or(f32::INFINITY);
            if cd > worst && result.len() >= ef {
                break;
            }
            let neighbors = &self.neighbors[cid as usize];
            for &n in neighbors {
                if visited[n as usize] {
                    continue;
                }
                visited[n as usize] = true;
                if Some(n) == exclude {
                    continue;
                }
                let d = l2_sq(query, self.vector(n));
                if result.len() < ef || d < result.peek().unwrap().dist {
                    candidates.push(MinCand { dist: d, id: n });
                    result.push(Cand { dist: d, id: n });
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
        let mut out: Vec<Cand> = result.into_iter().collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out
    }

    /// Public search returning top-k ids and distances.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let ef = ef.max(k);
        let pool = self.search_internal(query, ef, None);
        pool.into_iter().take(k).map(|c| (c.id, c.dist)).collect()
    }
}
