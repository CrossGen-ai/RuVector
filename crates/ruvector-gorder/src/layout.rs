//! Node-ID layouts. Each `Layout` computes a permutation `perm[old_id] = new_id`
//! and `apply_permutation` rewrites the graph in-place so that vectors and
//! adjacency lists are stored in the new order.
//!
//! Three implementations:
//!
//! * [`InsertionLayout`] — identity permutation (baseline).
//! * [`BfsLayout`]       — BFS traversal from `entry`; nodes are numbered in
//!   the order they are dequeued. Cheap, and already gives a big locality
//!   win for hub-rooted graphs.
//! * [`GorderLayout`]    — Gorder (Wei et al., SIGMOD 2016). A sliding-window
//!   greedy that, for each next position, picks the unassigned node maximizing
//!   `Sib(u,v) + Sfriend(u,v)` where Sib counts shared out-neighbors already
//!   placed within the window, and Sfriend counts direct edges to placed nodes
//!   within the window. This is O(n * w * m) with small constants and delivers
//!   the best cache locality of the three in our benches.

use crate::graph::MiniHnsw;
use serde::{Deserialize, Serialize};

pub trait Layout {
    /// Return `perm` such that `perm[old_id] = new_id`.
    fn permutation(&self, g: &MiniHnsw) -> Vec<u32>;
    fn name(&self) -> &'static str;
}

#[derive(Default, Copy, Clone)]
pub struct InsertionLayout;
impl Layout for InsertionLayout {
    fn permutation(&self, g: &MiniHnsw) -> Vec<u32> {
        (0..g.len() as u32).collect()
    }
    fn name(&self) -> &'static str {
        "insertion"
    }
}

#[derive(Default, Copy, Clone)]
pub struct BfsLayout;
impl Layout for BfsLayout {
    fn permutation(&self, g: &MiniHnsw) -> Vec<u32> {
        let n = g.len();
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut visited = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        q.push_back(g.entry);
        visited[g.entry as usize] = true;
        while let Some(id) = q.pop_front() {
            order.push(id);
            for &nb in &g.neighbors[id as usize] {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    q.push_back(nb);
                }
            }
        }
        // Any nodes disconnected from entry (rare in HNSW): append in id order.
        for i in 0..n {
            if !visited[i] {
                order.push(i as u32);
            }
        }
        // `order[new_id] = old_id` → invert.
        invert(&order)
    }
    fn name(&self) -> &'static str {
        "bfs"
    }
}

#[derive(Copy, Clone)]
pub struct GorderLayout {
    pub window: usize,
}
impl Default for GorderLayout {
    fn default() -> Self {
        Self { window: 8 }
    }
}

impl Layout for GorderLayout {
    fn permutation(&self, g: &MiniHnsw) -> Vec<u32> {
        let n = g.len();
        let w = self.window.max(1);

        // Build reverse-adjacency once (nodes that point TO u).
        let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (u, adj) in g.neighbors.iter().enumerate() {
            for &v in adj {
                rev[v as usize].push(u as u32);
            }
        }

        let mut placed = vec![false; n];
        let mut order: Vec<u32> = Vec::with_capacity(n);
        // Score of an unplaced node vs current window.
        let mut score = vec![0i32; n];

        // Seed: highest-out-degree node — a hub.
        let seed = (0..n).max_by_key(|&i| g.neighbors[i].len()).unwrap_or(0) as u32;
        order.push(seed);
        placed[seed as usize] = true;
        bump_scores(g, &rev, seed, &placed, &mut score, 1);

        while order.len() < n {
            // Pick the unplaced node with the highest score; break ties by degree.
            let mut best = -1i32;
            let mut best_id = u32::MAX;
            let mut best_deg = 0usize;
            for (i, &sc) in score.iter().enumerate() {
                if placed[i] {
                    continue;
                }
                let deg = g.neighbors[i].len();
                if sc as i32 > best || (sc as i32 == best && deg > best_deg) {
                    best = sc as i32;
                    best_id = i as u32;
                    best_deg = deg;
                }
            }
            if best_id == u32::MAX {
                // Should not happen; fall back to first unplaced.
                best_id = placed.iter().position(|&p| !p).unwrap() as u32;
            }
            order.push(best_id);
            placed[best_id as usize] = true;
            bump_scores(g, &rev, best_id, &placed, &mut score, 1);
            // Evict node leaving the window.
            if order.len() > w {
                let leaving = order[order.len() - 1 - w];
                bump_scores(g, &rev, leaving, &placed, &mut score, -1);
            }
        }
        invert(&order)
    }
    fn name(&self) -> &'static str {
        "gorder"
    }
}

#[inline]
fn bump_scores(
    g: &MiniHnsw,
    rev: &[Vec<u32>],
    node: u32,
    placed: &[bool],
    score: &mut [i32],
    delta: i32,
) {
    // Direct out-edges from `node` to unplaced -> boost (friend score).
    for &nb in &g.neighbors[node as usize] {
        if !placed[nb as usize] {
            score[nb as usize] += delta;
        }
    }
    // Reverse edges (in-neighbors of `node`) that are unplaced -> boost.
    for &nb in &rev[node as usize] {
        if !placed[nb as usize] {
            score[nb as usize] += delta;
        }
    }
    // Sibling score: nodes that share an out-neighbor with `node`.
    for &nb in &g.neighbors[node as usize] {
        for &sib in &rev[nb as usize] {
            if !placed[sib as usize] && sib != node {
                score[sib as usize] += delta;
            }
        }
    }
}

fn invert(order: &[u32]) -> Vec<u32> {
    let n = order.len();
    let mut perm = vec![0u32; n];
    for (new_id, &old_id) in order.iter().enumerate() {
        perm[old_id as usize] = new_id as u32;
    }
    perm
}

/// Rewrite the graph so node `old_id` becomes `perm[old_id]`.
/// Vectors are gathered into a new contiguous buffer in the new order.
pub fn apply_permutation(g: &MiniHnsw, perm: &[u32]) -> MiniHnsw {
    let n = g.len();
    let dim = g.dim();
    assert_eq!(perm.len(), n);

    // inverse: new_id -> old_id
    let mut inv = vec![0u32; n];
    for (old, &new_id) in perm.iter().enumerate() {
        inv[new_id as usize] = old as u32;
    }

    let mut new_vectors = vec![0f32; n * dim];
    let mut new_neighbors: Vec<Vec<u32>> = vec![Vec::new(); n];
    for new_id in 0..n {
        let old_id = inv[new_id] as usize;
        new_vectors[new_id * dim..new_id * dim + dim]
            .copy_from_slice(&g.vectors[old_id * dim..old_id * dim + dim]);
        new_neighbors[new_id] = g.neighbors[old_id]
            .iter()
            .map(|&nb| perm[nb as usize])
            .collect();
    }
    MiniHnsw {
        params: g.params.clone(),
        vectors: new_vectors,
        neighbors: new_neighbors,
        entry: perm[g.entry as usize],
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct LayoutStats {
    pub name: String,
    /// Average |new_id[u] - new_id[v]| over all edges — lower means better locality.
    pub mean_edge_span: f64,
    /// Fraction of edges with |new_id[u]-new_id[v]| <= 64 (roughly one cache line
    /// of 64B / f32=4B ⇒ 16 vectors of dim 64 fit in ~4KB, so 64 IDs is a
    /// proxy for "same L1 4KB block"). Higher is better.
    pub near_edge_frac: f64,
}

pub fn layout_stats(g: &MiniHnsw, perm: &[u32], name: &str) -> LayoutStats {
    let mut total: u64 = 0;
    let mut count: u64 = 0;
    let mut near: u64 = 0;
    for (u, adj) in g.neighbors.iter().enumerate() {
        let nu = perm[u] as i64;
        for &v in adj {
            let nv = perm[v as usize] as i64;
            let d = (nu - nv).unsigned_abs();
            total += d;
            count += 1;
            if d <= 64 {
                near += 1;
            }
        }
    }
    let count = count.max(1);
    LayoutStats {
        name: name.to_string(),
        mean_edge_span: total as f64 / count as f64,
        near_edge_frac: near as f64 / count as f64,
    }
}
