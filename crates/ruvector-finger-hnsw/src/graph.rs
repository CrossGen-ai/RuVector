//! Flat kNN graph: each node has `m` nearest neighbors by L2.
//!
//! Not full HNSW — the FINGER sketch mechanism is identical whether the
//! underlying graph is HNSW or any proximity graph, and a flat kNN graph
//! lets us measure the sketch's effect in isolation without the layered
//! HNSW machinery distorting the counts.

use crate::data::l2_sq;

#[derive(Clone, Debug)]
pub struct GraphParams {
    /// Out-degree per node (nearest-neighbor slots).
    pub m: usize,
    /// Random long-range shortcut edges per node. Replaces the last
    /// `m_long` neighbor slots with uniformly random ids, giving the
    /// graph small-world connectivity so a flat kNN graph does not get
    /// trapped inside a single cluster during best-first search.
    pub m_long: usize,
    /// Seed used for shortcut sampling and entry-point selection.
    pub seed: u64,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self { m: 16, m_long: 4, seed: 0xC0FFEE }
    }
}

#[derive(Clone, Debug)]
pub struct FlatGraph {
    pub dim: usize,
    pub n: usize,
    pub m: usize,
    /// Flat neighbor table, `n * m` indices.
    pub neighbors: Vec<u32>,
    /// A deterministic entry point for search.
    pub entry: u32,
}

impl FlatGraph {
    /// Build by brute-force kNN + random long-range shortcuts.
    /// Deterministic. O(n^2 * d) for the kNN pass.
    pub fn build(data: &[Vec<f32>], params: &GraphParams) -> Self {
        let n = data.len();
        assert!(n > 0, "empty dataset");
        let dim = data[0].len();
        let m = params.m.min(n.saturating_sub(1));
        let m_long = params.m_long.min(m);
        let m_near = m - m_long;

        let mut neighbors = vec![0u32; n * m];
        let mut rng = crate::data::Rng::new(params.seed ^ 0xA5A5A5A5A5A5A5A5);
        for i in 0..n {
            // Nearest `m_near` neighbors.
            let mut scored: Vec<(u32, f32)> = (0..n as u32)
                .filter(|&j| j as usize != i)
                .map(|j| (j, l2_sq(&data[i], &data[j as usize])))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.truncate(m_near);
            for (k, (id, _)) in scored.iter().enumerate() {
                neighbors[i * m + k] = *id;
            }
            // Random long-range shortcuts (dedup against near neighbors).
            let mut filled = m_near;
            let mut guard = 0;
            while filled < m && guard < m_long * 10 {
                let r = (rng.unit() * n as f32) as u32 % n as u32;
                if r as usize == i { guard += 1; continue; }
                let exists = (0..filled).any(|k| neighbors[i * m + k] == r);
                if !exists {
                    neighbors[i * m + filled] = r;
                    filled += 1;
                }
                guard += 1;
            }
            // Pad any remaining slots with self-ring (rare in practice).
            while filled < m {
                neighbors[i * m + filled] = ((i + filled + 1) % n) as u32;
                filled += 1;
            }
        }

        let entry = pick_entry(data, params.seed);
        Self { dim, n, m, neighbors, entry }
    }

    #[inline]
    pub fn neighbors_of(&self, node: u32) -> &[u32] {
        let s = node as usize * self.m;
        &self.neighbors[s..s + self.m]
    }
}

fn pick_entry(data: &[Vec<f32>], seed: u64) -> u32 {
    // Pick medoid among the first `sample` nodes with a cheap hash-seed shuffle.
    let n = data.len();
    let sample = 32.min(n);
    let mut best = 0u32;
    let mut best_cost = f32::INFINITY;
    for i in 0..sample {
        let idx = ((seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(i as u64 * 7919)) as usize) % n;
        let mut cost = 0.0f32;
        for j in 0..sample {
            let jdx = ((seed.wrapping_add(j as u64).wrapping_mul(0xBF58476D1CE4E5B9)) as usize) % n;
            cost += l2_sq(&data[idx], &data[jdx]);
        }
        if cost < best_cost {
            best_cost = cost;
            best = idx as u32;
        }
    }
    best
}
