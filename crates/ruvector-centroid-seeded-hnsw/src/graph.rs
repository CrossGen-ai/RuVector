//! k-NN graph + greedy best-first search (proxy for HNSW top-layer descent).

use crate::sqdist;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy, Default)]
pub struct SearchStats {
    pub hops: usize,
    pub distance_calls: usize,
}

pub struct KnnGraph {
    pub data: Vec<Vec<f32>>,
    pub adj: Vec<Vec<u32>>, // out-neighbors per node
}

impl KnnGraph {
    /// Build a symmetric k-NN graph in O(N^2 * D). PoC-scale only.
    pub fn build(data: Vec<Vec<f32>>, k: usize) -> Self {
        let n = data.len();
        let mut adj: Vec<Vec<u32>> = vec![Vec::with_capacity(k * 2); n];
        for i in 0..n {
            // Find k nearest excluding self.
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(k + 1);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = sqdist(&data[i], &data[j]);
                // Use f32 bits as ordering key (works because dists are >= 0).
                let key = d.to_bits();
                if heap.len() < k {
                    heap.push((key, j as u32));
                } else if let Some(top) = heap.peek() {
                    if key < top.0 {
                        heap.pop();
                        heap.push((key, j as u32));
                    }
                }
            }
            for (_, j) in heap {
                adj[i].push(j);
            }
        }
        // Symmetrize.
        let mut sym = adj.clone();
        for i in 0..n {
            for &j in &adj[i] {
                if !sym[j as usize].contains(&(i as u32)) {
                    sym[j as usize].push(i as u32);
                }
            }
        }
        Self { data, adj: sym }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Greedy best-first search: descend from each entry until no neighbor improves.
    /// Returns the best node found and stats.
    pub fn greedy_search(&self, q: &[f32], entries: &[usize]) -> (usize, SearchStats) {
        let mut stats = SearchStats::default();
        let mut visited = vec![false; self.data.len()];
        let mut best = entries[0];
        let mut best_d = sqdist(&self.data[best], q);
        stats.distance_calls += 1;
        visited[best] = true;
        // Seed multi-entry: pick globally best of entries first.
        for &e in entries.iter().skip(1) {
            if visited[e] {
                continue;
            }
            visited[e] = true;
            let d = sqdist(&self.data[e], q);
            stats.distance_calls += 1;
            if d < best_d {
                best_d = d;
                best = e;
            }
        }
        loop {
            let mut improved = false;
            let mut next = best;
            let mut next_d = best_d;
            for &nb in &self.adj[best] {
                let nb = nb as usize;
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let d = sqdist(&self.data[nb], q);
                stats.distance_calls += 1;
                if d < next_d {
                    next_d = d;
                    next = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
            best = next;
            best_d = next_d;
            stats.hops += 1;
        }
        (best, stats)
    }
}
