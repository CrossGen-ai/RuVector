//! Simplified single-layer NSW-like graph builder with reciprocated edges.
//!
//! For each newly inserted point `u`, run an ef=`ef_construction` beam search
//! against the current graph, keep the `m_max` closest as forward neighbors,
//! then also add reciprocal `v -> u` edges (subject to `m_max`, replacing the
//! furthest neighbor of `v` if full and `u` is closer).
//!
//! Reciprocation is essential — without it, only the seed nodes are ever
//! reachable from the entry point, and recall collapses.

use crate::graph::{l2_sq, BidirGraph};

#[derive(Debug, Clone, Copy)]
pub struct BuildParams {
    pub m_max: usize,
    pub ef_construction: usize,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self { m_max: 16, ef_construction: 64 }
    }
}

pub fn build_hnsw_like(vectors: &[Vec<f32>], p: BuildParams) -> BidirGraph {
    let n = vectors.len();
    let mut g = BidirGraph::empty(n, p.m_max);
    g.entry = 0;
    if n < 2 {
        return g;
    }
    // seed with a single edge 0 <-> 1.
    g.set_neighbor(0, 0, 1);
    g.set_neighbor(1, 0, 0);

    for u in 2..n {
        let entry = g.entry;
        let candidates =
            beam_search_build(&g, vectors, &vectors[u], entry, p.ef_construction);
        // Diversity-prune (HNSW's select_neighbors_heuristic).
        let selected = select_neighbors_heuristic(&candidates, vectors, p.m_max);
        for (slot, &(_, v)) in selected.iter().enumerate() {
            g.set_neighbor(u, slot, v);
        }
        // reciprocate: add u -> v back-edges into v's list, re-pruning if full.
        for &(_, v) in &selected {
            add_reciprocal_pruned(&mut g, vectors, v, u);
        }
    }
    g
}

/// HNSW-style diversity heuristic: greedy pick from sorted candidates; keep `v`
/// only if it is closer to the query (`u`'s vector — encoded implicitly via
/// each candidate's distance) than to any already-selected neighbor.
fn select_neighbors_heuristic(
    candidates: &[(f32, usize)],
    vectors: &[Vec<f32>],
    m_max: usize,
) -> Vec<(f32, usize)> {
    let mut selected: Vec<(f32, usize)> = Vec::with_capacity(m_max);
    for &(d_uv, v) in candidates {
        if selected.len() >= m_max {
            break;
        }
        let mut keep = true;
        for &(_, w) in &selected {
            let d_vw = l2_sq(&vectors[v], &vectors[w]);
            if d_vw < d_uv {
                // v is closer to an already-selected neighbor than to u →
                // redundant; skip to preserve diversity.
                keep = false;
                break;
            }
        }
        if keep {
            selected.push((d_uv, v));
        }
    }
    // If diversity pruning gave < m_max, top up with next-closest candidates.
    if selected.len() < m_max {
        for &(d, v) in candidates {
            if selected.len() >= m_max {
                break;
            }
            if !selected.iter().any(|(_, w)| *w == v) {
                selected.push((d, v));
            }
        }
    }
    selected
}

fn add_reciprocal_pruned(g: &mut BidirGraph, vectors: &[Vec<f32>], v: usize, u: usize) {
    if v == u {
        return;
    }
    let base = v * g.m_max;
    // already present?
    for slot in 0..g.m_max {
        if g.adj[base + slot] == u {
            return;
        }
        if g.adj[base + slot] == usize::MAX {
            g.adj[base + slot] = u;
            return;
        }
    }
    // v's list is full. Re-run diversity heuristic over {existing neighbors ∪ u}.
    let mut cand: Vec<(f32, usize)> = (0..g.m_max)
        .map(|slot| g.adj[base + slot])
        .filter(|&w| w != usize::MAX)
        .map(|w| (l2_sq(&vectors[v], &vectors[w]), w))
        .collect();
    cand.push((l2_sq(&vectors[v], &vectors[u]), u));
    cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let selected = select_neighbors_heuristic(&cand, vectors, g.m_max);
    for slot in 0..g.m_max {
        g.adj[base + slot] = selected.get(slot).map(|x| x.1).unwrap_or(usize::MAX);
    }
}

/// Return sorted ascending (dist_sq, id), length ≤ ef.
fn beam_search_build(
    g: &BidirGraph,
    vectors: &[Vec<f32>],
    q: &[f32],
    entry: usize,
    ef: usize,
) -> Vec<(f32, usize)> {
    let n = g.n;
    let mut visited = vec![false; n];
    let entry_d = l2_sq(q, &vectors[entry]);
    visited[entry] = true;
    let mut frontier: Vec<(f32, usize)> = vec![(entry_d, entry)];
    let mut best: Vec<(f32, usize)> = vec![(entry_d, entry)];
    while let Some((d_u, u)) = pop_min(&mut frontier) {
        let worst = best.last().map(|x| x.0).unwrap_or(f32::MAX);
        if d_u > worst && best.len() >= ef {
            break;
        }
        for v in g.neighbors(u) {
            if visited[v] {
                continue;
            }
            visited[v] = true;
            let d_v = l2_sq(q, &vectors[v]);
            let worst_now = best.last().map(|x| x.0).unwrap_or(f32::MAX);
            if best.len() < ef || d_v < worst_now {
                frontier.push((d_v, v));
                insert_sorted(&mut best, (d_v, v), ef);
            }
        }
    }
    best
}

fn pop_min(v: &mut Vec<(f32, usize)>) -> Option<(f32, usize)> {
    if v.is_empty() {
        return None;
    }
    let mut min_i = 0;
    for i in 1..v.len() {
        if v[i].0 < v[min_i].0 {
            min_i = i;
        }
    }
    Some(v.swap_remove(min_i))
}

fn insert_sorted(list: &mut Vec<(f32, usize)>, item: (f32, usize), cap: usize) {
    let pos = list.partition_point(|x| x.0 <= item.0);
    list.insert(pos, item);
    if list.len() > cap {
        list.truncate(cap);
    }
}
