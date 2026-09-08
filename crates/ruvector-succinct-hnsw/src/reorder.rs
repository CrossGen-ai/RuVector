//! BFS reordering: produces a permutation `old_to_new[i] = new_id` such
//! that graph-adjacent nodes land close in id space, which shrinks
//! successive-difference deltas fed into VarByte.
//!
//! Multi-source BFS from an evenly-spaced seed set covers disconnected
//! components. Ties broken by original id for determinism.

use crate::NodeId;
use std::collections::VecDeque;

/// Return `old_to_new` permutation. `lists[i]` is neighbours of node `i`
/// (in original id space).
pub fn bfs_permutation(lists: &[Vec<NodeId>], seeds: usize) -> Vec<NodeId> {
    let n = lists.len();
    let mut visited = vec![false; n];
    let mut old_to_new = vec![NodeId::MAX; n];
    let mut next_id: NodeId = 0;
    let mut queue: VecDeque<NodeId> = VecDeque::new();

    // Seed at evenly-spaced original ids so disconnected components are
    // guaranteed some starting point.
    let stride = (n / seeds.max(1)).max(1);
    let mut seed_iter = (0..n).step_by(stride);

    while (next_id as usize) < n {
        // Pull the next unvisited seed.
        let start = loop {
            match seed_iter.next() {
                Some(s) if !visited[s] => break s,
                Some(_) => continue,
                None => {
                    // Fallback: scan for any unvisited node.
                    match (0..n).find(|&i| !visited[i]) {
                        Some(s) => break s,
                        None => return old_to_new,
                    }
                }
            }
        };
        visited[start] = true;
        old_to_new[start] = next_id;
        next_id += 1;
        queue.push_back(start as NodeId);

        while let Some(u) = queue.pop_front() {
            // Sort neighbours by original id for deterministic ordering.
            let mut nbrs = lists[u as usize].clone();
            nbrs.sort_unstable();
            for v in nbrs {
                let vi = v as usize;
                if !visited[vi] {
                    visited[vi] = true;
                    old_to_new[vi] = next_id;
                    next_id += 1;
                    queue.push_back(v);
                }
            }
        }
    }
    old_to_new
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_a_permutation() {
        let lists: Vec<Vec<NodeId>> = vec![
            vec![1, 2],
            vec![0, 3],
            vec![0, 4],
            vec![1],
            vec![2],
        ];
        let p = bfs_permutation(&lists, 1);
        let mut seen = vec![false; p.len()];
        for &new_id in &p {
            assert!(!seen[new_id as usize], "duplicate new id");
            seen[new_id as usize] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn disconnected_components_covered() {
        // Two disconnected pairs.
        let lists: Vec<Vec<NodeId>> = vec![vec![1], vec![0], vec![3], vec![2]];
        let p = bfs_permutation(&lists, 2);
        let mut sorted = p.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }
}
