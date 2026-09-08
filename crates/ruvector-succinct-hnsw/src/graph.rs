//! Deterministic single-layer NSW graph builder.
//!
//! We build the topology once as a plain `Vec<Vec<NodeId>>` and hand it
//! off to whichever [`crate::adjacency::Adjacency`] backend the benchmark
//! is measuring. This guarantees that every backend serves the *same*
//! graph — the only variable is how neighbour lists are stored/decoded.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::{Distance, NodeId, Vector};

/// Reverse-ordered f32 for max-heap-behaves-as-min-heap use.
#[derive(Clone, Copy, Debug)]
struct MinF32(f32, NodeId);

impl PartialEq for MinF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MinF32 {}
impl Ord for MinF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smaller distance = greater priority.
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}
impl PartialOrd for MinF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-heap over f32 (used for the working "best so far" set).
#[derive(Clone, Copy, Debug)]
struct MaxF32(f32, NodeId);

impl PartialEq for MaxF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxF32 {}
impl Ord for MaxF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
            .then(self.1.cmp(&other.1))
    }
}
impl PartialOrd for MaxF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct BuildParams {
    pub m: usize,               // max neighbours per node
    pub ef_construction: usize, // beam width during insertion
}

/// Build a proximity graph. Each node's neighbour list is its exact
/// `M`-nearest-neighbour set (excluding itself) — a "brute kNN graph".
/// This isolates the encoding-vs-baseline comparison from HNSW-builder
/// idiosyncrasies. Cost is O(N² · dim) which is fine for the sizes we
/// benchmark (N ≤ 20 k). `ef_construction` is retained in the API for
/// interface compatibility but is unused by this builder.
pub fn build_graph<D: Distance>(
    corpus: &[Vector],
    dist: &D,
    p: BuildParams,
) -> Vec<Vec<NodeId>> {
    let n = corpus.len();
    let mut lists: Vec<Vec<NodeId>> = Vec::with_capacity(n);
    if n == 0 {
        return lists;
    }
    let _ = p.ef_construction;
    for v in 0..n {
        let mut d: Vec<(f32, NodeId)> = (0..n as NodeId)
            .filter(|&u| u as usize != v)
            .map(|u| (dist.dist(&corpus[v], &corpus[u as usize]), u))
            .collect();
        let m_idx = p.m.min(d.len()) - 1;
        d.select_nth_unstable_by(m_idx, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        });
        let nbrs: Vec<NodeId> = d.into_iter().take(p.m).map(|(_, u)| u).collect();
        lists.push(nbrs);
    }
    // Symmetrise: add reverse edges to guarantee connectivity of the
    // undirected view (kNN alone can produce weakly-connected components
    // in clustered corpora). Then cap each node at 2M neighbours.
    let mut extra: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for v in 0..n {
        for &u in &lists[v] {
            extra[u as usize].push(v as NodeId);
        }
    }
    for v in 0..n {
        lists[v].extend(extra[v].drain(..));
        lists[v].sort_unstable();
        lists[v].dedup();
        let cap = 2 * p.m;
        if lists[v].len() > cap {
            // Keep the cap closest.
            let mut with_d: Vec<(f32, NodeId)> = lists[v]
                .iter()
                .map(|&u| (dist.dist(&corpus[v], &corpus[u as usize]), u))
                .collect();
            with_d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            with_d.truncate(cap);
            lists[v].clear();
            lists[v].extend(with_d.into_iter().map(|(_, u)| u));
            lists[v].sort_unstable();
        }
    }
    lists
}

fn connect_bidirectional<D: Distance>(
    v: NodeId,
    u: NodeId,
    d_vu: f32,
    lists: &mut [Vec<NodeId>],
    corpus: &[Vector],
    dist: &D,
    m: usize,
) {
    let ul = &mut lists[u as usize];
    ul.push(v);
    if ul.len() > m {
        // Prune to m closest to u.
        let mut with_d: Vec<(f32, NodeId)> = ul
            .iter()
            .map(|&x| (dist.dist(&corpus[u as usize], &corpus[x as usize]), x))
            .collect();
        with_d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        with_d.truncate(m);
        ul.clear();
        ul.extend(with_d.into_iter().map(|(_, x)| x));
    }
    let _ = d_vu;
}

/// Beam search restricted to the graph currently in `lists`. Returns
/// candidates sorted by ascending distance to `q`.
fn search_layer<D: Distance>(
    q: &[f32],
    corpus: &[Vector],
    lists: &[Vec<NodeId>],
    entry: NodeId,
    ef: usize,
    dist: &D,
) -> Vec<(f32, NodeId)> {
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut frontier: BinaryHeap<MinF32> = BinaryHeap::new();
    let mut best: BinaryHeap<MaxF32> = BinaryHeap::new();
    let d0 = dist.dist(q, &corpus[entry as usize]);
    visited.insert(entry);
    frontier.push(MinF32(d0, entry));
    best.push(MaxF32(d0, entry));
    while let Some(MinF32(dc, c)) = frontier.pop() {
        if let Some(&MaxF32(worst, _)) = best.peek() {
            if best.len() >= ef && dc > worst {
                break;
            }
        }
        for &nb in &lists[c as usize] {
            if !visited.insert(nb) {
                continue;
            }
            let d = dist.dist(q, &corpus[nb as usize]);
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
    let mut out: Vec<(f32, NodeId)> =
        best.into_iter().map(|MaxF32(d, id)| (d, id)).collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SqEuclid;

    #[test]
    fn tiny_graph_builds() {
        let corpus: Vec<Vector> = (0..20).map(|i| vec![i as f32, (i * 2) as f32]).collect();
        let lists = build_graph(
            &corpus,
            &SqEuclid,
            BuildParams { m: 4, ef_construction: 16 },
        );
        assert_eq!(lists.len(), 20);
        // Every non-root node has at least one neighbour.
        for v in 1..lists.len() {
            assert!(!lists[v].is_empty());
        }
    }
}
