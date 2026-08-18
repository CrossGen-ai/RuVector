//! Greedy beam search over the graph. Layout-agnostic.

use crate::graph::{l2_sq, HnswGraph};
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

#[derive(Default, Clone, Debug)]
pub struct SearchStats {
    pub distances_computed: u64,
    pub nodes_visited: u64,
    /// Sum of |id(cur) - id(prev)| across visits. A coarse locality proxy —
    /// smaller means the visit sequence stays inside nearby memory pages.
    pub id_stride_sum: u64,
}

pub fn search_knn(g: &HnswGraph, q: &[f32], k: usize, ef: usize) -> (Vec<u32>, SearchStats) {
    let mut stats = SearchStats::default();
    let n = g.n();
    let mut visited = vec![false; n];
    let ep = g.entry;
    let d0 = l2_sq(q, g.vector(ep));
    stats.distances_computed += 1;
    let mut best = BinaryHeap::<Cand>::new(); // max-heap bounded by ef
    best.push(Cand { d: d0, id: ep });
    let mut cand = BinaryHeap::<Cand>::new(); // as min via neg
    cand.push(Cand { d: -d0, id: ep });
    visited[ep as usize] = true;

    let mut last_id: i64 = ep as i64;
    while let Some(c) = cand.pop() {
        let cd = -c.d;
        let worst = best.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
        if best.len() >= ef && cd > worst {
            break;
        }
        stats.nodes_visited += 1;
        stats.id_stride_sum += (c.id as i64 - last_id).unsigned_abs();
        last_id = c.id as i64;
        for &nb in g.neighbours_of(c.id) {
            let nu = nb as usize;
            if visited[nu] {
                continue;
            }
            visited[nu] = true;
            let dd = l2_sq(q, g.vector(nb));
            stats.distances_computed += 1;
            let worst = best.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
            if best.len() < ef || dd < worst {
                best.push(Cand { d: dd, id: nb });
                cand.push(Cand { d: -dd, id: nb });
                if best.len() > ef {
                    best.pop();
                }
            }
        }
    }

    let mut v: Vec<Cand> = best.into_iter().collect();
    v.sort_by(|a, b| a.d.partial_cmp(&b.d).unwrap_or(std::cmp::Ordering::Equal));
    v.truncate(k);
    (v.into_iter().map(|c| c.id).collect(), stats)
}
