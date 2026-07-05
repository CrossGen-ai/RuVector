//! A minimal, single-layer HNSW-style proximity graph used purely as a
//! **substrate** for measuring the effect of node layout on cache misses.
//!
//! We deliberately keep this simple (single layer, fixed M, greedy build)
//! because our research question is orthogonal to HNSW quality: given the
//! *same* graph, does reordering node IDs change search throughput?
//!
//! Vectors are stored in a contiguous `Vec<f32>` (row-major) so that a
//! permutation of node IDs directly changes the memory access pattern
//! during search.

use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MiniHnswParams {
    pub dim: usize,
    pub m: usize,           // neighbors per node
    pub ef_construction: usize,
    pub seed: u64,
}

impl Default for MiniHnswParams {
    fn default() -> Self {
        Self { dim: 64, m: 16, ef_construction: 64, seed: 42 }
    }
}

/// A single-layer proximity graph. `neighbors[i]` are the IDs of node i's
/// out-edges, and `vectors[i * dim ..][.. dim]` is its vector.
#[derive(Clone)]
pub struct MiniHnsw {
    pub params: MiniHnswParams,
    pub vectors: Vec<f32>,
    pub neighbors: Vec<Vec<u32>>,
    pub entry: u32,
}

impl MiniHnsw {
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.neighbors.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.params.dim
    }
    #[inline]
    pub fn vector(&self, id: u32) -> &[f32] {
        let d = self.params.dim;
        let s = id as usize * d;
        &self.vectors[s..s + d]
    }

    /// Build a random-vector graph with greedy connect. Deterministic given `seed`.
    pub fn build_random(n: usize, params: MiniHnswParams) -> Self {
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(params.seed);
        let normal = Normal::new(0.0_f32, 1.0_f32).unwrap();
        let dim = params.dim;
        let mut vectors = vec![0.0f32; n * dim];
        for v in vectors.iter_mut() {
            *v = normal.sample(&mut rng);
        }
        // Normalize (unit vectors — cosine == inner product == negated L2/2)
        for i in 0..n {
            let s = i * dim;
            let norm: f32 = vectors[s..s + dim].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for v in &mut vectors[s..s + dim] {
                *v /= norm;
            }
        }

        let mut g = MiniHnsw {
            params: params.clone(),
            vectors,
            neighbors: vec![Vec::with_capacity(params.m); n],
            entry: 0,
        };

        // Insert nodes one by one using a bounded greedy search against the
        // partial graph. Randomly seed the first node.
        for id in 1..n as u32 {
            let candidates = g.search_partial(id as usize, params.ef_construction);
            // Take the top-m closest as bidirectional neighbors.
            for &nb in candidates.iter().take(params.m) {
                g.neighbors[id as usize].push(nb);
                if g.neighbors[nb as usize].len() < params.m {
                    g.neighbors[nb as usize].push(id);
                } else {
                    // prune-by-distance: keep closest m
                    let mut ns = g.neighbors[nb as usize].clone();
                    ns.push(id);
                    ns.sort_by(|&a, &b| {
                        let da = l2sq(g.vector(a), g.vector(nb));
                        let db = l2sq(g.vector(b), g.vector(nb));
                        da.partial_cmp(&db).unwrap()
                    });
                    ns.truncate(params.m);
                    g.neighbors[nb as usize] = ns;
                }
            }
        }
        // Choose a hub-ish entry point: node with the highest in-degree.
        let mut indeg = vec![0u32; n];
        for adj in &g.neighbors {
            for &nb in adj {
                indeg[nb as usize] += 1;
            }
        }
        g.entry = indeg
            .iter()
            .enumerate()
            .max_by_key(|(_, &d)| d)
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        // Nudge in a tiny randomness so ties aren't always node 0.
        if rng.gen_bool(0.25) {
            g.entry = ((g.entry as usize + 1) % n) as u32;
        }
        g
    }

    /// Greedy search against the partial graph containing nodes [0, up_to).
    /// Used only during construction. Query is the vector at index `up_to`.
    fn search_partial(&self, up_to: usize, ef: usize) -> Vec<u32> {
        use std::collections::BinaryHeap;
        let query = self.vector(up_to as u32).to_vec();
        let start = if up_to == 0 { 0 } else { (up_to - 1) as u32 };
        let mut visited = vec![false; up_to.max(1)];
        // max-heap of (dist, id) trimmed to ef
        let mut best: BinaryHeap<Ordered> = BinaryHeap::new();
        // min-heap frontier
        let mut frontier: BinaryHeap<std::cmp::Reverse<Ordered>> = BinaryHeap::new();
        let d0 = l2sq(&query, self.vector(start));
        visited[start as usize] = true;
        best.push(Ordered(d0, start));
        frontier.push(std::cmp::Reverse(Ordered(d0, start)));
        while let Some(std::cmp::Reverse(Ordered(d, id))) = frontier.pop() {
            if let Some(worst) = best.peek() {
                if d > worst.0 && best.len() >= ef {
                    break;
                }
            }
            for &nb in &self.neighbors[id as usize] {
                if (nb as usize) >= up_to {
                    continue;
                }
                if visited[nb as usize] {
                    continue;
                }
                visited[nb as usize] = true;
                let dd = l2sq(&query, self.vector(nb));
                if best.len() < ef {
                    best.push(Ordered(dd, nb));
                    frontier.push(std::cmp::Reverse(Ordered(dd, nb)));
                } else if dd < best.peek().unwrap().0 {
                    best.pop();
                    best.push(Ordered(dd, nb));
                    frontier.push(std::cmp::Reverse(Ordered(dd, nb)));
                }
            }
        }
        let mut out: Vec<Ordered> = best.into_iter().collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.into_iter().map(|Ordered(_, id)| id).collect()
    }
}

/// Squared L2 distance. Kept simple; branch-free autovectorization
/// applies for the small dims we bench (64/128/256).
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) struct Ordered(pub f32, pub u32);
impl Eq for Ordered {}
impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap on distance.
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}
