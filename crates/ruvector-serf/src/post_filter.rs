//! Post-filter NSW: run greedy graph search on the full graph, then drop
//! results whose attribute falls outside the query range. This is the naive
//! baseline used by most production ANN systems when range filtering was
//! bolted on after the fact.

use std::collections::{BinaryHeap, HashSet};

use crate::data::{in_range, sq_l2, Dataset, Query};
use crate::graph::Graph;
use crate::{Hit, RangeAnn};

pub struct PostFilterNsw<'a> {
    pub ds: &'a Dataset,
    pub graph: Graph,
    pub ef: usize,
    pub entry: u32,
}

impl<'a> PostFilterNsw<'a> {
    pub fn new(ds: &'a Dataset, graph: Graph, ef: usize) -> Self {
        // Entry point: the medoid-ish node closest to the centroid. For random
        // gaussian data the centroid is ~origin, so node 0 is fine in practice;
        // pick id 0 to keep behavior reproducible.
        Self { ds, graph, ef, entry: 0 }
    }
}

/// Min-heap by distance (we negate so BinaryHeap-as-max works as a min-heap).
#[derive(Copy, Clone, PartialEq)]
struct Cand {
    dist: f32,
    id: u32,
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.dist.partial_cmp(&self.dist).unwrap()
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Standard NSW greedy search: maintain a candidate min-heap and a result
/// max-heap of size `ef`, expand until no improvement.
pub(crate) fn nsw_search(
    points: &[crate::data::Point],
    graph: &Graph,
    entry: u32,
    query: &Query,
    ef: usize,
    edge_filter: impl Fn(f32) -> bool,
) -> Vec<(f32, u32)> {
    let mut visited = HashSet::with_capacity(ef * 4);
    let mut candidates: BinaryHeap<Cand> = BinaryHeap::new();

    let d0 = sq_l2(&points[entry as usize].vec, &query.vec);
    candidates.push(Cand { dist: d0, id: entry });
    visited.insert(entry);
    let mut best: Vec<(f32, u32)> = vec![(d0, entry)];

    while let Some(cur) = candidates.pop() {
        // Threshold: stop when current candidate is worse than the worst
        // accepted result and we already have `ef` results.
        if best.len() >= ef {
            let worst = best
                .iter()
                .map(|&(d, _)| d)
                .fold(f32::NEG_INFINITY, f32::max);
            if cur.dist > worst {
                break;
            }
        }
        for &nbr in &graph.neighbors[cur.id as usize] {
            if !visited.insert(nbr) {
                continue;
            }
            let p_attr = points[nbr as usize].attr;
            // edge_filter prunes traversal in SeRF; pass-through for post-filter.
            if !edge_filter(p_attr) {
                continue;
            }
            let d = sq_l2(&points[nbr as usize].vec, &query.vec);
            // Decide whether to push into results.
            if best.len() < ef {
                best.push((d, nbr));
                candidates.push(Cand { dist: d, id: nbr });
            } else {
                let (worst_idx, worst_dist) = best
                    .iter()
                    .enumerate()
                    .map(|(i, &(d, _))| (i, d))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                if d < worst_dist {
                    best[worst_idx] = (d, nbr);
                    candidates.push(Cand { dist: d, id: nbr });
                }
            }
        }
    }

    best.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    best
}

impl<'a> RangeAnn for PostFilterNsw<'a> {
    fn name(&self) -> &'static str {
        "post-filter-nsw"
    }

    fn search(&self, query: &Query, k: usize) -> Vec<Hit> {
        // No edge filter — explore the whole graph, then filter results.
        let raw = nsw_search(
            &self.ds.points,
            &self.graph,
            self.entry,
            query,
            self.ef,
            |_attr| true,
        );
        let mut filtered: Vec<(f32, u32)> = raw
            .into_iter()
            .filter(|&(_, id)| {
                in_range(
                    self.ds.points[id as usize].attr,
                    query.lo,
                    query.hi,
                )
            })
            .collect();
        filtered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        filtered.truncate(k);
        filtered
            .into_iter()
            .map(|(d, id)| Hit { id, dist: d })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_filter_returns_in_range_only() {
        let ds = Dataset::random_gaussian(500, 8, 5);
        let g = Graph::build_knn(&ds.points, 8);
        let idx = PostFilterNsw::new(&ds, g, 32);
        let q = Query { vec: vec![0.0; 8], lo: 200.0, hi: 299.0 };
        let hits = idx.search(&q, 10);
        for h in &hits {
            let a = ds.points[h.id as usize].attr;
            assert!(a >= 200.0 && a <= 299.0);
        }
    }
}
