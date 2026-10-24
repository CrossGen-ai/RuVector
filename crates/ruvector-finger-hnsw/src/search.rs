//! Best-first graph search with three distance backends:
//!
//!  - `ExactSearch`   : standard, exact L2 at every candidate.
//!  - `FingerSearch`  : approximate distance from FINGER sketch only.
//!  - `FingerGuarded` : approximate as a lower-bound filter; recompute
//!                      exact L2 only when the approximation could beat
//!                      the current kth-best.
//!
//! All three share the same traversal shell so differences in result sets
//! come purely from the distance oracle.

use std::collections::BinaryHeap;

use crate::data::l2_sq;
use crate::graph::FlatGraph;
use crate::sketch::EdgeSketch;

#[derive(Clone, Debug, Default)]
pub struct SearchStats {
    pub exact_dist_calls: u64,
    pub approx_dist_calls: u64,
    pub visited: u64,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub ids: Vec<u32>,
    pub dists: Vec<f32>,
    pub stats: SearchStats,
}

pub trait Search {
    fn search(&self, data: &[Vec<f32>], graph: &FlatGraph, q: &[f32], k: usize, ef: usize) -> SearchResult;
}

/// Max-heap entry on candidate dist (min via Reverse).
#[derive(Copy, Clone, PartialEq)]
struct Hd {
    dist: f32,
    id: u32,
}
impl Eq for Hd {}
impl PartialOrd for Hd {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Hd {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // NaN-safe: treat NaN as largest.
        self.dist.partial_cmp(&o.dist).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn visited_set(n: usize) -> Vec<bool> {
    vec![false; n]
}

/// Exact baseline best-first search.
pub struct ExactSearch;
impl Search for ExactSearch {
    fn search(&self, data: &[Vec<f32>], graph: &FlatGraph, q: &[f32], k: usize, ef: usize) -> SearchResult {
        let n = graph.n;
        let mut stats = SearchStats::default();
        let mut visited = visited_set(n);

        let entry = graph.entry;
        let d_entry = l2_sq(&data[entry as usize], q);
        stats.exact_dist_calls += 1;
        visited[entry as usize] = true;

        // Candidate heap: pop smallest (so store Reverse). Result heap: pop largest.
        let mut cand = BinaryHeap::<std::cmp::Reverse<Hd>>::new();
        let mut res = BinaryHeap::<Hd>::new();
        cand.push(std::cmp::Reverse(Hd { dist: d_entry, id: entry }));
        res.push(Hd { dist: d_entry, id: entry });

        while let Some(std::cmp::Reverse(top)) = cand.pop() {
            // Current worst in results.
            let worst = res.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
            if top.dist > worst && res.len() >= ef {
                break;
            }
            stats.visited += 1;
            for &nbr in graph.neighbors_of(top.id) {
                if visited[nbr as usize] {
                    continue;
                }
                visited[nbr as usize] = true;
                let d = l2_sq(&data[nbr as usize], q);
                stats.exact_dist_calls += 1;
                let worst_now = res.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
                if res.len() < ef || d < worst_now {
                    res.push(Hd { dist: d, id: nbr });
                    if res.len() > ef {
                        res.pop();
                    }
                    cand.push(std::cmp::Reverse(Hd { dist: d, id: nbr }));
                }
            }
        }

        finalize(res, k, stats)
    }
}

/// Pure sketch search: approximate distance decides everything.
pub struct FingerSearch<'a> {
    pub sketch: &'a EdgeSketch,
}
impl<'a> Search for FingerSearch<'a> {
    fn search(&self, data: &[Vec<f32>], graph: &FlatGraph, q: &[f32], k: usize, ef: usize) -> SearchResult {
        let n = graph.n;
        let mut stats = SearchStats::default();
        let mut visited = visited_set(n);
        let dim = graph.dim;

        let entry = graph.entry;
        let d_entry = l2_sq(&data[entry as usize], q);
        stats.exact_dist_calls += 1;
        visited[entry as usize] = true;

        let mut cand = BinaryHeap::<std::cmp::Reverse<Hd>>::new();
        let mut res = BinaryHeap::<Hd>::new();
        cand.push(std::cmp::Reverse(Hd { dist: d_entry, id: entry }));
        res.push(Hd { dist: d_entry, id: entry });

        let mut delta = vec![0.0f32; dim];
        let mut delta_proj = vec![0.0f32; self.sketch.k];

        while let Some(std::cmp::Reverse(top)) = cand.pop() {
            let worst = res.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
            if top.dist > worst && res.len() >= ef {
                break;
            }
            stats.visited += 1;

            // Compute delta and its projection once per visited node.
            let c_vec = &data[top.id as usize];
            for i in 0..dim {
                delta[i] = q[i] - c_vec[i];
            }
            self.sketch.project_delta(&delta, &mut delta_proj);

            let m = graph.m;
            for slot in 0..m {
                let nbr = graph.neighbors_of(top.id)[slot];
                if visited[nbr as usize] {
                    continue;
                }
                visited[nbr as usize] = true;

                let ip = self.sketch.approx_ip(&delta_proj, self.sketch.edge_slot_row(top.id, slot));
                let sq_len = self.sketch.edge_slot_sqlen(top.id, slot);
                let approx = (top.dist + sq_len - 2.0 * ip).max(0.0);
                stats.approx_dist_calls += 1;

                let worst_now = res.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
                if res.len() < ef || approx < worst_now {
                    res.push(Hd { dist: approx, id: nbr });
                    if res.len() > ef {
                        res.pop();
                    }
                    cand.push(std::cmp::Reverse(Hd { dist: approx, id: nbr }));
                }
            }
        }

        // Final rerank: compute exact distance over the ef results so
        // output distances are honest even in pure-sketch mode.
        let rough = finalize(res, ef, stats.clone());
        rerank_exact(data, q, rough, k, stats)
    }
}

/// Guarded: approximate is used strictly as a lower-bound filter.
///
/// For every neighbor we compute `approx - margin` as a cheap lower bound.
/// If that lower bound is already worse than the current kth-best, we skip
/// the exact L2. Otherwise we fall back to exact. The margin absorbs
/// JL-approximation variance.
pub struct FingerGuarded<'a> {
    pub sketch: &'a EdgeSketch,
    /// Lower-bound slack as a fraction of the current kth-best (e.g. 0.10
    /// = allow 10% headroom before trusting the lower bound to filter).
    pub slack: f32,
}
impl<'a> Search for FingerGuarded<'a> {
    fn search(&self, data: &[Vec<f32>], graph: &FlatGraph, q: &[f32], k: usize, ef: usize) -> SearchResult {
        let n = graph.n;
        let mut stats = SearchStats::default();
        let mut visited = visited_set(n);
        let dim = graph.dim;

        let entry = graph.entry;
        let d_entry = l2_sq(&data[entry as usize], q);
        stats.exact_dist_calls += 1;
        visited[entry as usize] = true;

        let mut cand = BinaryHeap::<std::cmp::Reverse<Hd>>::new();
        let mut res = BinaryHeap::<Hd>::new();
        cand.push(std::cmp::Reverse(Hd { dist: d_entry, id: entry }));
        res.push(Hd { dist: d_entry, id: entry });

        let mut delta = vec![0.0f32; dim];
        let mut delta_proj = vec![0.0f32; self.sketch.k];

        while let Some(std::cmp::Reverse(top)) = cand.pop() {
            let worst = res.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
            if top.dist > worst && res.len() >= ef {
                break;
            }
            stats.visited += 1;

            let c_vec = &data[top.id as usize];
            for i in 0..dim {
                delta[i] = q[i] - c_vec[i];
            }
            self.sketch.project_delta(&delta, &mut delta_proj);

            let m = graph.m;
            for slot in 0..m {
                let nbr = graph.neighbors_of(top.id)[slot];
                if visited[nbr as usize] {
                    continue;
                }
                visited[nbr as usize] = true;

                let ip = self.sketch.approx_ip(&delta_proj, self.sketch.edge_slot_row(top.id, slot));
                let sq_len = self.sketch.edge_slot_sqlen(top.id, slot);
                let approx = (top.dist + sq_len - 2.0 * ip).max(0.0);
                stats.approx_dist_calls += 1;

                let worst_now = res.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
                let threshold = worst_now * (1.0 + self.slack);

                // If even the generous approximation can't beat threshold,
                // skip exact. Else fall through to exact.
                if res.len() >= ef && approx > threshold {
                    continue;
                }

                let d = l2_sq(&data[nbr as usize], q);
                stats.exact_dist_calls += 1;

                let worst_now = res.peek().map(|h| h.dist).unwrap_or(f32::INFINITY);
                if res.len() < ef || d < worst_now {
                    res.push(Hd { dist: d, id: nbr });
                    if res.len() > ef {
                        res.pop();
                    }
                    cand.push(std::cmp::Reverse(Hd { dist: d, id: nbr }));
                }
            }
        }

        finalize(res, k, stats)
    }
}

fn finalize(mut heap: BinaryHeap<Hd>, k: usize, stats: SearchStats) -> SearchResult {
    let mut items: Vec<Hd> = Vec::with_capacity(heap.len());
    while let Some(h) = heap.pop() {
        items.push(h);
    }
    items.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
    items.truncate(k);
    let ids = items.iter().map(|h| h.id).collect();
    let dists = items.iter().map(|h| h.dist).collect();
    SearchResult { ids, dists, stats }
}

/// Rerank ef approximate results with exact L2 and return top-k by exact.
fn rerank_exact(data: &[Vec<f32>], q: &[f32], mut rough: SearchResult, k: usize, mut stats: SearchStats) -> SearchResult {
    for (i, id) in rough.ids.iter().enumerate() {
        rough.dists[i] = l2_sq(&data[*id as usize], q);
        stats.exact_dist_calls += 1;
    }
    let mut pairs: Vec<(u32, f32)> = rough
        .ids
        .iter()
        .zip(rough.dists.iter())
        .map(|(a, b)| (*a, *b))
        .collect();
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(k);
    SearchResult {
        ids: pairs.iter().map(|p| p.0).collect(),
        dists: pairs.iter().map(|p| p.1).collect(),
        stats,
    }
}
