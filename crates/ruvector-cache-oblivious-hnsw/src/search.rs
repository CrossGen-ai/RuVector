//! Greedy top-k search on a permuted `FlatGraph`. Layout-agnostic — every
//! variant runs the same code path; only the memory locality of `vector()` /
//! `neighbours_of()` reads differs.

use crate::build::l2_sq;
use crate::graph::FlatGraph;

#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub k: usize,
    pub ef: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchStats {
    pub visited: u64,
    pub distance_evals: u64,
}

/// Returns top-k `(dist_sq, slot)` closest to `query`, together with counters.
pub fn greedy_search(
    graph: &FlatGraph,
    query: &[f32],
    params: SearchParams,
) -> (Vec<(f32, u32)>, SearchStats) {
    let n = graph.n();
    let mut visited = vec![false; n];
    let mut stats = SearchStats::default();

    let entry = graph.entry;
    let ev = graph.vector(entry);
    let d0 = l2_sq(ev, query);
    stats.distance_evals += 1;
    visited[entry as usize] = true;
    stats.visited += 1;

    let mut top: Vec<(f32, u32)> = vec![(d0, entry)];
    let mut cand: Vec<(f32, u32)> = vec![(d0, entry)];

    while let Some((cd, cid)) = pop_min(&mut cand) {
        let worst = top.last().map(|x| x.0).unwrap_or(f32::INFINITY);
        if cd > worst && top.len() >= params.ef { break; }
        for &nb in graph.neighbours_of(cid) {
            if nb == u32::MAX { continue; }
            if visited[nb as usize] { continue; }
            visited[nb as usize] = true;
            stats.visited += 1;
            let nv = graph.vector(nb);
            let d = l2_sq(nv, query);
            stats.distance_evals += 1;
            let worst = top.last().map(|x| x.0).unwrap_or(f32::INFINITY);
            if top.len() < params.ef || d < worst {
                push_sorted(&mut top, (d, nb), params.ef);
                cand.push((d, nb));
            }
        }
    }
    top.truncate(params.k);
    (top, stats)
}

fn pop_min(v: &mut Vec<(f32, u32)>) -> Option<(f32, u32)> {
    if v.is_empty() { return None; }
    let mut best = 0;
    for i in 1..v.len() {
        if v[i].0 < v[best].0 { best = i; }
    }
    Some(v.swap_remove(best))
}

fn push_sorted(top: &mut Vec<(f32, u32)>, item: (f32, u32), cap: usize) {
    let pos = top.iter().position(|x| x.0 > item.0).unwrap_or(top.len());
    top.insert(pos, item);
    if top.len() > cap { top.truncate(cap); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{build_hnsw, BuildParams};
    use crate::graph::Layout;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn params_small() -> BuildParams {
        BuildParams { n: 800, m: 16, ef_construction: 32, seed: 42 }
    }

    #[test]
    fn build_all_three_layouts_and_search_returns_something() {
        let base = params_small();
        for layout in [Layout::Bfs, Layout::Dfs, Layout::Veb] {
            let g = build_hnsw(base, layout);
            assert_eq!(g.n(), base.n);
            assert_eq!(g.m, base.m);
            // Query with a stored vector — must find it in top-1 with high prob.
            let q = g.vector(g.perm[7]).to_vec();
            let (top, _) = greedy_search(&g, &q, SearchParams { k: 5, ef: 64 });
            assert!(!top.is_empty());
        }
    }

    #[test]
    fn permutation_is_a_bijection_all_layouts() {
        let base = params_small();
        for layout in [Layout::Bfs, Layout::Dfs, Layout::Veb] {
            let g = build_hnsw(base, layout);
            let mut seen = vec![false; g.n()];
            for &s in &g.perm { seen[s as usize] = true; }
            assert!(seen.iter().all(|&x| x), "perm not bijective for {:?}", g.layout);
        }
    }

    #[test]
    fn layouts_return_identical_top1_for_stored_queries() {
        // The graphs share build params, so exact recall for stored vectors
        // should match across layouts (up to ordering of ties).
        let base = params_small();
        let g_bfs = build_hnsw(base, Layout::Bfs);
        let g_dfs = build_hnsw(base, Layout::Dfs);
        let g_veb = build_hnsw(base, Layout::Veb);
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let mut matches = 0;
        for _ in 0..30 {
            let id: u32 = rng.gen_range(0..base.n as u32);
            // Query the *logical* vector (same across all layouts).
            let q = g_bfs.vector(g_bfs.perm[id as usize]).to_vec();
            let (t_b, _) = greedy_search(&g_bfs, &q, SearchParams { k: 1, ef: 64 });
            let (t_d, _) = greedy_search(&g_dfs, &q, SearchParams { k: 1, ef: 64 });
            let (t_v, _) = greedy_search(&g_veb, &q, SearchParams { k: 1, ef: 64 });
            // Map slot → logical id for comparison.
            let lb = g_bfs.inv[t_b[0].1 as usize];
            let ld = g_dfs.inv[t_d[0].1 as usize];
            let lv = g_veb.inv[t_v[0].1 as usize];
            if lb == ld && ld == lv { matches += 1; }
        }
        assert!(matches >= 25, "layouts diverged on stored queries: {}/30", matches);
    }
}
