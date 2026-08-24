//! Node ordering strategies.

use crate::{FlatGraph, NodeOrdering};
use std::collections::VecDeque;

/// Identity ordering (insertion order = build order). Baseline.
pub struct Insertion;
impl NodeOrdering for Insertion {
    fn permute(&self, g: &FlatGraph) -> Vec<u32> {
        (0..g.n as u32).collect()
    }
    fn name(&self) -> &'static str { "insertion" }
}

/// Breadth-first layout starting from the graph entry point.
/// New id = BFS visit order → co-visited neighbors land in adjacent cache lines.
pub struct Bfs;
impl NodeOrdering for Bfs {
    fn permute(&self, g: &FlatGraph) -> Vec<u32> {
        let n = g.n;
        let mut perm = vec![u32::MAX; n];
        let mut visited = vec![false; n];
        let mut queue: VecDeque<u32> = VecDeque::with_capacity(n);
        let mut next_id: u32 = 0;

        queue.push_back(g.entry);
        visited[g.entry as usize] = true;
        while let Some(u) = queue.pop_front() {
            perm[u as usize] = next_id;
            next_id += 1;
            let base = u as usize * g.max_degree;
            let c = g.neighbor_counts[u as usize] as usize;
            for s in 0..c {
                let v = g.neighbors[base + s] as usize;
                if v < n && !visited[v] {
                    visited[v] = true;
                    queue.push_back(v as u32);
                }
            }
        }
        // Unreachable nodes get appended in insertion order.
        for i in 0..n {
            if perm[i] == u32::MAX {
                perm[i] = next_id;
                next_id += 1;
            }
        }
        perm
    }
    fn name(&self) -> &'static str { "bfs" }
}

/// Reverse Cuthill–McKee ordering. Same as BFS but at each level, neighbors
/// are appended in ascending degree order — a classic bandwidth-reduction
/// heuristic for sparse matrices, and empirically the smallest edge-span
/// layout for HNSW-like graphs.
pub struct ReverseCuthillMcKee;
impl NodeOrdering for ReverseCuthillMcKee {
    fn permute(&self, g: &FlatGraph) -> Vec<u32> {
        let n = g.n;
        let deg = &g.neighbor_counts;
        // Start seed: lowest-degree unvisited node (repeat for disconnected parts).
        let mut visited = vec![false; n];
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut queue: VecDeque<u32> = VecDeque::with_capacity(n);

        // Seed by graph entry first (cache-warm start), then fill any islands.
        queue.push_back(g.entry);
        visited[g.entry as usize] = true;

        loop {
            while let Some(u) = queue.pop_front() {
                order.push(u);
                let base = u as usize * g.max_degree;
                let c = deg[u as usize] as usize;
                let mut children: Vec<u32> = (0..c)
                    .map(|s| g.neighbors[base + s])
                    .filter(|&v| (v as usize) < n && !visited[v as usize])
                    .collect();
                // Ascending degree.
                children.sort_by_key(|&v| deg[v as usize]);
                for v in children {
                    if !visited[v as usize] {
                        visited[v as usize] = true;
                        queue.push_back(v);
                    }
                }
            }
            // any remaining? seed lowest-degree unvisited.
            let mut best: Option<(u32, u32)> = None;
            for i in 0..n {
                if !visited[i] {
                    let d = deg[i];
                    match best {
                        None => best = Some((i as u32, d)),
                        Some((_, bd)) if d < bd => best = Some((i as u32, d)),
                        _ => {}
                    }
                }
            }
            match best {
                None => break,
                Some((u, _)) => {
                    visited[u as usize] = true;
                    queue.push_back(u);
                }
            }
        }
        // Reverse to get RCM (empirically better locality than plain CM).
        order.reverse();
        let mut perm = vec![0u32; n];
        for (new_id, &old_id) in order.iter().enumerate() {
            perm[old_id as usize] = new_id as u32;
        }
        perm
    }
    fn name(&self) -> &'static str { "reverse-cuthill-mckee" }
}
