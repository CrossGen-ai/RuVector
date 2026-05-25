//! Minimal single-level Navigable Small World graph.
//!
//! Single-level (no HNSW hierarchy). Insertion connects to `m` approximate
//! nearest neighbors via beam search with `ef_construction` candidates.
//! Search uses standard greedy beam-search. Self-contained, no external deps.
//!
//! The graph shares vector storage via [`Arc`] so segment-tree variants can
//! build many graphs over disjoint id subsets without duplicating data.

use crate::sq_l2;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct NswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for NswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 64,
            ef_search: 32,
        }
    }
}

pub struct Nsw {
    pub vectors: Arc<Vec<Vec<f32>>>,
    pub ids: Vec<u32>,
    adj: Vec<Vec<u32>>,
    entry: Option<u32>,
    params: NswParams,
}

#[derive(Copy, Clone, Debug)]
struct Cand {
    id: u32,
    dist: f32,
}
impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

impl Nsw {
    pub fn build(vectors: Arc<Vec<Vec<f32>>>, ids: Vec<u32>, params: NswParams) -> Self {
        let n = ids.len();
        let mut s = Self {
            vectors,
            ids,
            adj: vec![Vec::new(); n],
            entry: None,
            params,
        };
        for i in 0..n {
            s.insert(i as u32);
        }
        s
    }

    #[inline]
    fn vec_of(&self, local: u32) -> &[f32] {
        &self.vectors[self.ids[local as usize] as usize]
    }

    fn insert(&mut self, new_local: u32) {
        let entry = match self.entry {
            None => {
                self.entry = Some(new_local);
                return;
            }
            Some(e) => e,
        };
        let q = self.vec_of(new_local).to_vec();
        let neighbors = self.search_layer(&q, entry, self.params.ef_construction, Some(new_local));
        let m = self.params.m.min(neighbors.len());
        for &Cand { id, .. } in neighbors.iter().take(m) {
            self.adj[new_local as usize].push(id);
            self.adj[id as usize].push(new_local);
            if self.adj[id as usize].len() > 2 * self.params.m {
                let me = self.vec_of(id).to_vec();
                let mut nbrs: Vec<Cand> = self.adj[id as usize]
                    .iter()
                    .map(|&n| Cand {
                        id: n,
                        dist: sq_l2(&me, self.vec_of(n)),
                    })
                    .collect();
                nbrs.sort();
                nbrs.truncate(self.params.m);
                self.adj[id as usize] = nbrs.into_iter().map(|c| c.id).collect();
            }
        }
        if self.adj[new_local as usize].is_empty() {
            self.adj[new_local as usize].push(entry);
        }
    }

    fn search_layer(&self, q: &[f32], start: u32, ef: usize, exclude: Option<u32>) -> Vec<Cand> {
        let start_dist = sq_l2(q, self.vec_of(start));
        let mut visited: HashSet<u32> = HashSet::new();
        visited.insert(start);
        let mut candidates: BinaryHeap<std::cmp::Reverse<Cand>> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();
        candidates.push(std::cmp::Reverse(Cand {
            id: start,
            dist: start_dist,
        }));
        if exclude != Some(start) {
            results.push(Cand {
                id: start,
                dist: start_dist,
            });
        }
        while let Some(std::cmp::Reverse(c)) = candidates.pop() {
            let worst = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
            if c.dist > worst && results.len() >= ef {
                break;
            }
            let nbrs = self.adj[c.id as usize].clone();
            for n in nbrs {
                if !visited.insert(n) {
                    continue;
                }
                let d = sq_l2(q, self.vec_of(n));
                let worst = results.peek().map(|x| x.dist).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(std::cmp::Reverse(Cand { id: n, dist: d }));
                    if exclude != Some(n) {
                        results.push(Cand { id: n, dist: d });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        let mut out: Vec<Cand> = results.into_iter().collect();
        out.sort();
        out
    }

    pub fn search(&self, q: &[f32], k: usize) -> Vec<(usize, f32)> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        let res = self.search_layer(q, entry, self.params.ef_search.max(k), None);
        res.into_iter()
            .take(k)
            .map(|c| (self.ids[c.id as usize] as usize, c.dist))
            .collect()
    }

    pub fn adj_bytes(&self) -> usize {
        self.adj.iter().map(|a| a.capacity() * 4).sum::<usize>() + self.adj.len() * 24
    }
}
