//! Backend-agnostic beam search over any [`Adjacency`] store.
//!
//! Returns the k nearest node ids for `q` along with the number of
//! distance evaluations performed (used to sanity-check that all backends
//! do the same *work* — memory should be the only free variable).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::adjacency::Adjacency;
use crate::{Distance, NodeId, Vector};

#[derive(Clone, Copy, Debug)]
struct MinF32(f32, NodeId);
impl PartialEq for MinF32 {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for MinF32 {}
impl Ord for MinF32 {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal).then(self.1.cmp(&o.1))
    }
}
impl PartialOrd for MinF32 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Clone, Copy, Debug)]
struct MaxF32(f32, NodeId);
impl PartialEq for MaxF32 {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for MaxF32 {}
impl Ord for MaxF32 {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal).then(self.1.cmp(&o.1))
    }
}
impl PartialOrd for MaxF32 {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

pub struct SearchResult {
    pub topk: Vec<(f32, NodeId)>,
    pub dist_evals: usize,
}

pub fn beam_search<A: Adjacency, D: Distance>(
    q: &[f32],
    corpus: &[Vector],
    adj: &A,
    dist: &D,
    entry: NodeId,
    ef: usize,
    k: usize,
) -> SearchResult {
    beam_search_multi(q, corpus, adj, dist, &[entry], ef, k)
}

/// Multi-entry variant: seeds the frontier with several starting nodes
/// (useful when the graph has weak long-range connectivity — the beam
/// still needs to hop, but starts closer on average).
pub fn beam_search_multi<A: Adjacency, D: Distance>(
    q: &[f32],
    corpus: &[Vector],
    adj: &A,
    dist: &D,
    entries: &[NodeId],
    ef: usize,
    k: usize,
) -> SearchResult {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut frontier: BinaryHeap<MinF32> = BinaryHeap::new();
    let mut best: BinaryHeap<MaxF32> = BinaryHeap::new();
    let mut evals = 0usize;
    let mut nbrs: Vec<NodeId> = Vec::with_capacity(64);

    for &entry in entries {
        if !visited.insert(entry) {
            continue;
        }
        let d0 = dist.dist(q, &corpus[entry as usize]);
        evals += 1;
        frontier.push(MinF32(d0, entry));
        best.push(MaxF32(d0, entry));
        if best.len() > ef {
            best.pop();
        }
    }

    while let Some(MinF32(dc, c)) = frontier.pop() {
        if let Some(&MaxF32(worst, _)) = best.peek() {
            if best.len() >= ef && dc > worst {
                break;
            }
        }
        adj.neighbors_into(c, &mut nbrs);
        for &nb in &nbrs {
            if !visited.insert(nb) {
                continue;
            }
            let d = dist.dist(q, &corpus[nb as usize]);
            evals += 1;
            if best.len() < ef {
                frontier.push(MinF32(d, nb));
                best.push(MaxF32(d, nb));
            } else if let Some(&MaxF32(worst, _)) = best.peek() {
                if d < worst {
                    frontier.push(MinF32(d, nb));
                    best.push(MaxF32(d, nb));
                    if best.len() > ef {
                        best.pop();
                    }
                }
            }
        }
    }
    let mut topk: Vec<(f32, NodeId)> =
        best.into_iter().map(|MaxF32(d, id)| (d, id)).collect();
    topk.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    topk.truncate(k);
    SearchResult { topk, dist_evals: evals }
}
