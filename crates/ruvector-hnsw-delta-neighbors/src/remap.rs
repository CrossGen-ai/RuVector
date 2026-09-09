//! Locality-preserving ID remapping.
//!
//! Given a graph on IDs `0..n`, produce a permutation `perm` such that
//! `perm[old_id] = new_id` and graph neighbors receive numerically close new
//! IDs. This is what makes sorted-delta encodings pay off: small deltas fit
//! in one varint byte or a few bit-packed bits.
//!
//! The algorithm is a plain BFS from an arbitrary seed, with neighbors
//! visited in ascending old-ID order (stable across runs). BFS on a k-NN
//! graph mimics the "reverse Cuthill-McKee" ordering used to reduce sparse
//! matrix bandwidth; the same principle applies to neighbor-list compression.

use crate::graph::Graph;
use std::collections::VecDeque;

/// Build a BFS permutation. `perm[old] = new`. Nodes unreachable from the
/// seed are appended at the end in ascending old-ID order (so isolated
/// components do not silently drop out).
pub fn locality_remap_bfs(graph: &Graph, seed: u32) -> Vec<u32> {
    let n = graph.len();
    let mut perm = vec![u32::MAX; n];
    let mut counter: u32 = 0;
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(seed.min((n as u32).saturating_sub(1)));
    while let Some(u) = queue.pop_front() {
        if perm[u as usize] != u32::MAX {
            continue;
        }
        perm[u as usize] = counter;
        counter += 1;
        for &v in &graph.adj[u as usize] {
            if perm[v as usize] == u32::MAX {
                queue.push_back(v);
            }
        }
    }
    // Append any unreachable nodes.
    for i in 0..n {
        if perm[i] == u32::MAX {
            perm[i] = counter;
            counter += 1;
        }
    }
    debug_assert_eq!(counter as usize, n);
    perm
}

/// Apply a permutation to `graph` in place, producing a new graph whose IDs
/// are renumbered. Neighbor lists remain sorted-ascending.
pub fn apply_permutation(graph: &Graph, perm: &[u32]) -> Graph {
    let n = graph.len();
    assert_eq!(perm.len(), n);
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (old_u, row) in graph.adj.iter().enumerate() {
        let new_u = perm[old_u] as usize;
        adj[new_u] = row.iter().map(|&v| perm[v as usize]).collect();
        adj[new_u].sort_unstable();
    }
    Graph { adj }
}

/// Sum of |delta| across every (sorted) neighbor list. Diagnostic; smaller
/// after a good remap.
pub fn total_delta_magnitude(graph: &Graph) -> u64 {
    let mut s: u64 = 0;
    for row in &graph.adj {
        let mut prev: i64 = -1;
        for &v in row {
            s += (v as i64 - prev) as u64;
            prev = v as i64;
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{brute_knn_graph, random_vectors};

    #[test]
    fn remap_reduces_delta_magnitude() {
        let v = random_vectors(256, 16, 7);
        let g = brute_knn_graph(&v, 256, 16, 12);
        let before = total_delta_magnitude(&g);
        let perm = locality_remap_bfs(&g, 0);
        let g2 = apply_permutation(&g, &perm);
        let after = total_delta_magnitude(&g2);
        // BFS remap should reduce total delta magnitude on a k-NN graph.
        assert!(after < before, "after={} before={}", after, before);
    }

    #[test]
    fn permutation_is_bijection() {
        let v = random_vectors(64, 4, 3);
        let g = brute_knn_graph(&v, 64, 4, 6);
        let perm = locality_remap_bfs(&g, 0);
        let mut seen = vec![false; 64];
        for &p in &perm {
            assert!(!seen[p as usize]);
            seen[p as usize] = true;
        }
    }
}
