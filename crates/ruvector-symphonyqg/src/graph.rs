//! Small-world graph builder, NSW-style (a flat single-layer variant of
//! HNSW). Neighbors are selected with full-precision distance during build;
//! query-time traversal uses whatever distance functor is provided.
//!
//! This is intentionally a single layer — the goal of the PoC is to
//! isolate the effect of quantized-distance traversal, not to reproduce
//! the entire HNSW hierarchy.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Adjacency list. `neighbors[i]` is the (capped) friend list of node `i`.
#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub m: usize,
    pub neighbors: Vec<Vec<u32>>,
}

/// Bounded max-heap entry used during search.
#[derive(Clone, Copy, Debug)]
struct DistNode { dist: f32, id: u32 }

impl PartialEq for DistNode { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for DistNode {}
impl Ord for DistNode {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for DistNode { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

/// Build an NSW graph. `dist(i, j)` returns the (full-precision) squared
/// L2 distance between nodes `i` and `j`. `m` is the target neighbor cap.
/// `ef_construction` is the candidate-list size during insertion.
pub fn build_nsw<F: Fn(u32, u32) -> f32>(
    n: usize,
    m: usize,
    ef_construction: usize,
    dist: F,
) -> Graph {
    let mut neighbors: Vec<Vec<u32>> = (0..n).map(|_| Vec::with_capacity(m)).collect();
    if n == 0 {
        return Graph { n, m, neighbors };
    }
    // Seed: connect first few nodes in a small clique.
    let seed = n.min(m + 1);
    for i in 0..seed {
        for j in 0..seed {
            if i != j { neighbors[i].push(j as u32); }
        }
        if neighbors[i].len() > m { neighbors[i].truncate(m); }
    }
    if n <= seed { return Graph { n, m, neighbors }; }

    // Insert the rest. Use the existing graph + `dist` callback to find
    // candidates, then prune to m using simple-but-effective heuristic
    // (closest first, no relative-neighborhood pruning — this is a PoC).
    for new_id in seed..n {
        // Greedy entry-point search starting from node 0.
        let mut visited = vec![false; n];
        visited[0] = true;
        let mut top_w: BinaryHeap<DistNode> = BinaryHeap::new(); // max-heap of best ef
        let mut candidates: BinaryHeap<std::cmp::Reverse<DistNode>> = BinaryHeap::new();

        let d0 = dist(new_id as u32, 0);
        top_w.push(DistNode { dist: d0, id: 0 });
        candidates.push(std::cmp::Reverse(DistNode { dist: d0, id: 0 }));

        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            let worst = top_w.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && top_w.len() >= ef_construction { break; }
            for &nb in &neighbors[c.id as usize] {
                if nb as usize >= new_id { continue; } // only look at already-inserted nodes
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let d = dist(new_id as u32, nb);
                if top_w.len() < ef_construction || d < top_w.peek().unwrap().dist {
                    top_w.push(DistNode { dist: d, id: nb });
                    candidates.push(std::cmp::Reverse(DistNode { dist: d, id: nb }));
                    if top_w.len() > ef_construction { top_w.pop(); }
                }
            }
        }

        // Extract sorted (ascending) candidate list.
        let mut sorted: Vec<DistNode> = top_w.into_sorted_vec();
        sorted.reverse(); // into_sorted_vec returns ascending? Doc says ascending for max-heap. Be safe.
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        // Pick top-m as neighbors of new node, and add bidirectional links
        // with degree cap on the other side.
        for (k, dn) in sorted.iter().enumerate() {
            if k >= m { break; }
            neighbors[new_id].push(dn.id);
            let other = dn.id as usize;
            if neighbors[other].len() < m {
                neighbors[other].push(new_id as u32);
            } else {
                // Replace the worst existing neighbor if the new edge is shorter.
                let mut worst_idx = 0usize;
                let mut worst_d = -1f32;
                for (i, &nb) in neighbors[other].iter().enumerate() {
                    let dd = dist(other as u32, nb);
                    if dd > worst_d { worst_d = dd; worst_idx = i; }
                }
                if dn.dist < worst_d {
                    neighbors[other][worst_idx] = new_id as u32;
                }
            }
        }
    }

    Graph { n, m, neighbors }
}

/// Greedy beam search. `dist(id)` returns the query-to-node distance under
/// whatever metric the caller chose (exact or quantized).
pub fn search<F: FnMut(u32) -> f32>(
    g: &Graph,
    entry: u32,
    ef: usize,
    k: usize,
    mut dist: F,
) -> Vec<(u32, f32)> {
    if g.n == 0 { return Vec::new(); }
    let mut visited = vec![false; g.n];
    let mut top: BinaryHeap<DistNode> = BinaryHeap::new();
    let mut cands: BinaryHeap<std::cmp::Reverse<DistNode>> = BinaryHeap::new();

    visited[entry as usize] = true;
    let d0 = dist(entry);
    top.push(DistNode { dist: d0, id: entry });
    cands.push(std::cmp::Reverse(DistNode { dist: d0, id: entry }));

    while let Some(std::cmp::Reverse(c)) = cands.pop() {
        let worst = top.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
        if c.dist > worst && top.len() >= ef { break; }
        for &nb in &g.neighbors[c.id as usize] {
            if visited[nb as usize] { continue; }
            visited[nb as usize] = true;
            let d = dist(nb);
            if top.len() < ef || d < top.peek().unwrap().dist {
                top.push(DistNode { dist: d, id: nb });
                cands.push(std::cmp::Reverse(DistNode { dist: d, id: nb }));
                if top.len() > ef { top.pop(); }
            }
        }
    }

    let mut out: Vec<DistNode> = top.into_sorted_vec();
    out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    out.into_iter().take(k).map(|d| (d.id, d.dist)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_connectivity_smoke() {
        let n = 32;
        let pts: Vec<Vec<f32>> = (0..n).map(|i| vec![i as f32, (i*i) as f32]).collect();
        let g = build_nsw(n, 6, 16, |a, b| {
            let (a, b) = (&pts[a as usize], &pts[b as usize]);
            (a[0]-b[0]).powi(2) + (a[1]-b[1]).powi(2)
        });
        // Every node has at least one neighbor.
        for i in 0..n {
            assert!(!g.neighbors[i].is_empty(), "node {} isolated", i);
        }
    }
}
