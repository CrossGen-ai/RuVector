//! Deterministic HNSW-lite builder.
//!
//! We construct a single-layer k-NN proximity graph by incremental
//! insertion with greedy nearest-neighbour search. This mirrors the
//! base-layer topology of HNSW well enough for locality experiments,
//! while keeping build time bounded and reproducible.

use crate::graph::{l2_sq, HnswGraph};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BinaryHeap;

#[derive(Copy, Clone, PartialEq)]
struct Cand {
    d: f32,
    id: u32,
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.d.partial_cmp(&other.d).unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Build an m-degree proximity graph over `data`.
/// `ef_construction` controls candidate-set size during insertion.
pub fn build_hnsw(data: Vec<f32>, dim: usize, m: usize, ef_construction: usize) -> HnswGraph {
    let n = data.len() / dim;
    assert!(n > 0);
    let vec_of = |i: u32| -> &[f32] {
        let s = i as usize * dim;
        &data[s..s + dim]
    };

    // adjacency being built (mutable Vec<Vec<u32>> then compacted)
    let mut adj: Vec<Vec<u32>> = vec![Vec::with_capacity(m); n];
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    let entry: u32 = 0;

    for i in 1..n {
        // Greedy search from entry using existing partial graph.
        let mut visited = vec![false; i];
        let start = if i > 1 { rng.gen_range(0..i as u32) } else { 0 };
        let mut ep = start;
        // frontier: min-heap of (dist, id) as max-heap via neg;
        // we keep a working set W of size ef.
        let q = vec_of(i as u32);
        let mut best = BinaryHeap::<Cand>::new(); // max-heap of best (bounded)
        let d0 = l2_sq(q, vec_of(ep));
        best.push(Cand { d: d0, id: ep });
        visited[ep as usize] = true;
        // candidate frontier: max-heap of negative dist (we'll invert)
        let mut cand = BinaryHeap::<Cand>::new();
        cand.push(Cand { d: -d0, id: ep });

        while let Some(c) = cand.pop() {
            let cd = -c.d;
            let top = best.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
            if best.len() >= ef_construction && cd > top {
                break;
            }
            for &nb in &adj[c.id as usize] {
                let nu = nb as usize;
                if nu >= i || visited[nu] {
                    continue;
                }
                visited[nu] = true;
                let dd = l2_sq(q, vec_of(nb));
                let worst = best.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
                if best.len() < ef_construction || dd < worst {
                    best.push(Cand { d: dd, id: nb });
                    cand.push(Cand { d: -dd, id: nb });
                    if best.len() > ef_construction {
                        best.pop();
                    }
                }
            }
            let _ = ep;
        }

        // Select top-m neighbours by distance.
        let mut neigh: Vec<Cand> = best.into_iter().collect();
        neigh.sort_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(std::cmp::Ordering::Equal));
        neigh.truncate(m);
        for c in &neigh {
            adj[i].push(c.id);
        }
        // bidirectional link (with degree cap)
        for c in &neigh {
            let u = c.id as usize;
            if adj[u].len() < m * 2 {
                adj[u].push(i as u32);
            }
        }
    }

    // Compact to CSR.
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0u32);
    let mut neighbours: Vec<u32> = Vec::new();
    for row in &adj {
        neighbours.extend_from_slice(row);
        offsets.push(neighbours.len() as u32);
    }

    HnswGraph {
        dim,
        data,
        offsets,
        neighbours,
        entry,
    }
}
