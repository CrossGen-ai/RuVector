//! Greedy best-first beam search over the HCNNG proximity graph.
//!
//! This is the same shape as HNSW's level-0 search: maintain a min-heap of
//! candidates to expand and a max-heap of the top-k found so far. Terminate
//! once the best candidate is farther than the worst top-k.

use crate::distance::Distance;
use crate::graph::Graph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub id: u32,
    pub distance: f32,
}

/// Min-heap entry by distance.
#[derive(Debug, Clone, Copy)]
struct CandMin {
    d: f32,
    id: u32,
}
impl PartialEq for CandMin {
    fn eq(&self, o: &Self) -> bool {
        self.d == o.d
    }
}
impl Eq for CandMin {}
impl PartialOrd for CandMin {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for CandMin {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is max-heap; invert to get min-heap by d.
        o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap entry by distance.
#[derive(Debug, Clone, Copy)]
struct CandMax {
    d: f32,
    id: u32,
}
impl PartialEq for CandMax {
    fn eq(&self, o: &Self) -> bool {
        self.d == o.d
    }
}
impl Eq for CandMax {}
impl PartialOrd for CandMax {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for CandMax {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.partial_cmp(&o.d).unwrap_or(Ordering::Equal)
    }
}

pub fn beam_search<D: Distance + ?Sized>(
    query: &[f32],
    graph: &Graph,
    vectors: &[Vec<f32>],
    dist: &D,
    entries: &[u32],
    k: usize,
    ef: usize,
    visited_scratch: &mut HashSet<u32>,
) -> Vec<SearchResult> {
    visited_scratch.clear();
    let mut candidates: BinaryHeap<CandMin> = BinaryHeap::new();
    let mut top: BinaryHeap<CandMax> = BinaryHeap::new();

    for &entry in entries {
        if !visited_scratch.insert(entry) {
            continue;
        }
        let d0 = dist.d(query, &vectors[entry as usize]);
        candidates.push(CandMin { d: d0, id: entry });
        top.push(CandMax { d: d0, id: entry });
        if top.len() > ef {
            top.pop();
        }
    }

    while let Some(c) = candidates.pop() {
        // Stop condition: closest candidate is farther than current kth best.
        let worst_top = top.peek().map(|t| t.d).unwrap_or(f32::INFINITY);
        if top.len() >= ef && c.d > worst_top {
            break;
        }

        for &nb in &graph.neighbors[c.id as usize] {
            if !visited_scratch.insert(nb) {
                continue;
            }
            let dnb = dist.d(query, &vectors[nb as usize]);
            let worst = top.peek().map(|t| t.d).unwrap_or(f32::INFINITY);
            if top.len() < ef || dnb < worst {
                candidates.push(CandMin { d: dnb, id: nb });
                top.push(CandMax { d: dnb, id: nb });
                if top.len() > ef {
                    top.pop();
                }
            }
        }
    }

    // Drain top into sorted results, take k.
    let mut out: Vec<SearchResult> = top
        .into_iter()
        .map(|c| SearchResult {
            id: c.id,
            distance: c.d,
        })
        .collect();
    out.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    out.truncate(k);
    out
}
