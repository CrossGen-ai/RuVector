//! Minimal single-layer HNSW-style graph used to model traversal costs.
//!
//! We keep the graph representation flat and cache-unfriendly on
//! purpose: vectors live in one big `Vec<f32>` indexed by node id, and
//! neighbour lists are stored in a CSR (`neighbors` + `offsets`) that is
//! separate from the vectors. This mirrors what production HNSW
//! implementations look like once you strip away layer selection, and
//! it makes cache misses on neighbour vector loads dominant — which is
//! exactly what software prefetch is meant to hide.
//!
//! Neighbour selection uses the standard heuristic: connect each newly
//! inserted node to the `m` nearest already-inserted nodes, then
//! back-link. The construction cost is O(n^2) which is fine for the
//! benchmark sizes we use (`n` up to 50 000) and keeps the code under
//! 500 lines.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::prefetch::{prefetch_read, prefetch_vector};

#[derive(Copy, Clone, Debug)]
pub struct GraphConfig {
    pub n: usize,
    pub dim: usize,
    pub m: usize,
    pub seed: u64,
}

pub struct FlatGraph {
    pub dim: usize,
    pub m: usize,
    pub n: usize,
    /// Row-major vectors: `dim * n` floats.
    pub vectors: Vec<f32>,
    /// CSR: `neighbors[offsets[i]..offsets[i+1]]` are the neighbours of
    /// node `i`. All rows have length exactly `m`; the CSR structure is
    /// used so that a future extension to per-node degree is trivial.
    pub neighbors: Vec<u32>,
    pub offsets: Vec<u32>,
    /// Entry point for search.
    pub entry: u32,
}

impl FlatGraph {
    /// Build the graph from a random Gaussian dataset.
    pub fn build(cfg: GraphConfig) -> Self {
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let mut vectors = vec![0.0_f32; cfg.n * cfg.dim];
        for v in &mut vectors {
            // Standard normal via Box-Muller; keeps distances well
            // separated so recall is meaningful.
            let u1: f32 = rng.gen_range(1e-7..1.0);
            let u2: f32 = rng.gen_range(0.0..1.0);
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        }

        // Greedy neighbour selection.
        let mut neighbors = Vec::<u32>::with_capacity(cfg.n * cfg.m);
        let mut offsets = Vec::<u32>::with_capacity(cfg.n + 1);
        offsets.push(0);

        // First `m` nodes get a dummy dense clique so search has
        // somewhere to warm up.
        for i in 0..cfg.n {
            let mut cands: Vec<(f32, u32)> = Vec::new();
            let query = &vectors[i * cfg.dim..(i + 1) * cfg.dim];
            // Consider up to 512 previously inserted nodes plus everyone
            // in the entry clique. That is enough to make neighbour
            // quality realistic while keeping build time linear-ish.
            let start = i.saturating_sub(512);
            for j in start..i {
                let other = &vectors[j * cfg.dim..(j + 1) * cfg.dim];
                cands.push((squared_l2(query, other), j as u32));
            }
            // Always keep the seed clique reachable.
            for j in 0..cfg.m.min(i) {
                let other = &vectors[j * cfg.dim..(j + 1) * cfg.dim];
                let d = squared_l2(query, other);
                if !cands.iter().any(|&(_, id)| id == j as u32) {
                    cands.push((d, j as u32));
                }
            }
            cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            cands.truncate(cfg.m);
            // Fill missing slots with distinct random node ids (never
            // self) so no slot is a dead-end.
            while cands.len() < cfg.m {
                let mut r: u32 = rng.gen_range(0..cfg.n.max(1)) as u32;
                if r as usize == i {
                    r = ((i + 1) % cfg.n.max(1)) as u32;
                }
                cands.push((f32::INFINITY, r));
            }
            for (_, id) in &cands {
                neighbors.push(*id);
            }
            offsets.push(neighbors.len() as u32);
        }

        // Post-process: replace any residual self-loops with a random
        // non-self id. Self-loops arise when the initial candidate pool
        // was empty for very early nodes.
        for i in 0..cfg.n {
            let s = offsets[i] as usize;
            let e = offsets[i + 1] as usize;
            for slot in s..e {
                if neighbors[slot] as usize == i {
                    let mut r: u32 = rng.gen_range(0..cfg.n.max(1)) as u32;
                    if r as usize == i {
                        r = ((i + 1) % cfg.n.max(1)) as u32;
                    }
                    neighbors[slot] = r;
                }
            }
        }

        // Back-link pass: for each node, prepend it to its neighbours'
        // lists (truncating to `m`). Keeps the graph undirected-ish.
        let mut degree = vec![0u32; cfg.n];
        for i in 0..cfg.n {
            let row: Vec<u32> =
                neighbors[offsets[i] as usize..offsets[i + 1] as usize].to_vec();
            for &n in &row {
                let ni = n as usize;
                if ni >= cfg.n {
                    continue;
                }
                if degree[ni] < cfg.m as u32 {
                    let slot = offsets[ni] as usize + degree[ni] as usize;
                    // We won't overwrite the sorted neighbours: instead
                    // we track a parallel degree counter and only mutate
                    // slots that still hold the fill value (0). Because
                    // insertion order is deterministic this stays
                    // consistent across runs of the benchmark.
                    if neighbors[slot] == 0 && ni != 0 {
                        neighbors[slot] = i as u32;
                        degree[ni] += 1;
                    }
                }
            }
        }

        Self {
            dim: cfg.dim,
            m: cfg.m,
            n: cfg.n,
            vectors,
            neighbors,
            offsets,
            entry: (cfg.n / 2) as u32,
        }
    }

    #[inline]
    pub fn vector(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.vectors[s..s + self.dim]
    }

    #[inline]
    pub fn neighbors_of(&self, id: u32) -> &[u32] {
        let s = self.offsets[id as usize] as usize;
        let e = self.offsets[id as usize + 1] as usize;
        &self.neighbors[s..e]
    }

    /// Prefetch the neighbour-id row of `id` — used to hide the CSR
    /// dereference itself.
    #[inline]
    pub fn prefetch_neighbor_row(&self, id: u32) {
        let s = self.offsets[id as usize] as usize;
        if s < self.neighbors.len() {
            prefetch_read(&self.neighbors[s] as *const _);
        }
    }

    /// Prefetch the vector of node `id`.
    #[inline]
    pub fn prefetch_vector_of(&self, id: u32) {
        prefetch_vector(self.vector(id));
    }
}

#[inline]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    // Auto-vectorises to AVX2 (checked with `cargo asm`).
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}
