//! Small HNSW-style graph builder.
//!
//! This is not a production HNSW: it's a single-layer proximity graph built
//! greedily by inserting points and connecting each to its `M` nearest
//! already-inserted neighbours. It's enough to exercise the search hot path
//! and reproduce the layout effect we want to measure.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::graph::{FlatGraph, Layout};
use crate::layout::{bfs_permutation, dfs_permutation, veb_permutation, Adj};
use crate::DIM;

#[derive(Debug, Clone, Copy)]
pub struct BuildParams {
    pub n: usize,
    pub m: usize,
    /// ef_construction — width of the candidate frontier during insert.
    pub ef_construction: usize,
    pub seed: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self { n: 20_000, m: 16, ef_construction: 64, seed: 0xC0FFEE }
    }
}

#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..DIM {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Build the graph in id order, then apply a layout permutation.
pub fn build_hnsw(params: BuildParams, layout: Layout) -> FlatGraph {
    let n = params.n;
    let m = params.m;
    let mut rng = ChaCha8Rng::seed_from_u64(params.seed);

    // Random vectors on the unit sphere (roughly): sample N(0,1) then normalize.
    let mut vectors = vec![0f32; n * DIM];
    for v in vectors.iter_mut() {
        // Approx N(0,1) via 12-uniform CLT — fine for research corpus.
        let mut acc = 0f32;
        for _ in 0..12 {
            acc += rng.gen::<f32>();
        }
        *v = acc - 6.0;
    }
    for i in 0..n {
        let s = &mut vectors[i * DIM..(i + 1) * DIM];
        let mut norm = 0f32;
        for &x in s.iter() { norm += x * x; }
        let inv = if norm > 0.0 { 1.0 / norm.sqrt() } else { 1.0 };
        for x in s.iter_mut() { *x *= inv; }
    }

    // Logical adjacency during build (Vec<u32> per node).
    let mut adj: Adj = vec![Vec::with_capacity(m); n];

    // Insert nodes one at a time. For node i > 0, greedy-search the current
    // graph from entry=0 for a frontier of size ef, pick top-M as neighbours,
    // add reverse edges (trimming to M when over).
    for i in 1..n {
        let q = &vectors[i * DIM..(i + 1) * DIM].to_vec();
        let neighbours = search_frontier(
            &vectors,
            &adj,
            /*entry=*/ 0,
            q,
            params.ef_construction.max(m),
        );
        let take = neighbours.iter().take(m).map(|&(_, id)| id).collect::<Vec<_>>();
        adj[i] = take.clone();
        for u in take {
            if adj[u as usize].len() < m {
                adj[u as usize].push(i as u32);
            } else {
                // Replace the worst existing neighbour if this new one is
                // closer — keeps degree bounded and graph quality reasonable.
                let ui = u as usize;
                let uv = &vectors[ui * DIM..(ui + 1) * DIM];
                let d_new = l2_sq(uv, q);
                let mut worst_idx = 0;
                let mut worst_d = f32::MIN;
                for (k, &nb) in adj[ui].iter().enumerate() {
                    let nb_v = &vectors[nb as usize * DIM..(nb as usize + 1) * DIM];
                    let d = l2_sq(uv, nb_v);
                    if d > worst_d { worst_d = d; worst_idx = k; }
                }
                if d_new < worst_d {
                    adj[ui][worst_idx] = i as u32;
                }
            }
        }
    }

    // Compute permutation over the finished logical adjacency.
    let entry_logical: u32 = 0;
    let perm = match layout {
        Layout::Bfs => bfs_permutation(n, entry_logical, &adj),
        Layout::Dfs => dfs_permutation(n, entry_logical, &adj),
        Layout::Veb => veb_permutation(n, entry_logical, &adj),
    };
    let mut inv = vec![0u32; n];
    for (id, &slot) in perm.iter().enumerate() { inv[slot as usize] = id as u32; }

    // Reorder vectors and neighbour lists into slot order.
    let mut new_vectors = vec![0f32; n * DIM];
    let mut new_neighbours = vec![u32::MAX; n * m];
    for slot in 0..n {
        let id = inv[slot] as usize;
        new_vectors[slot * DIM..(slot + 1) * DIM]
            .copy_from_slice(&vectors[id * DIM..(id + 1) * DIM]);
        let dst = &mut new_neighbours[slot * m..(slot + 1) * m];
        for (k, &nb_id) in adj[id].iter().enumerate() {
            dst[k] = perm[nb_id as usize];
        }
    }

    let entry_slot = perm[entry_logical as usize];
    FlatGraph {
        layout,
        vectors: new_vectors,
        neighbours: new_neighbours,
        m,
        perm,
        inv,
        entry: entry_slot,
    }
}

/// Small frontier greedy search used during build. Operates on logical ids
/// with a `Vec<Vec<u32>>` adjacency, so it does not depend on any layout.
fn search_frontier(
    vectors: &[f32],
    adj: &[Vec<u32>],
    entry: u32,
    q: &[f32],
    ef: usize,
) -> Vec<(f32, u32)> {
    let mut visited = vec![false; adj.len()];
    let entry_v = &vectors[entry as usize * DIM..(entry as usize + 1) * DIM];
    let d0 = l2_sq(entry_v, q);
    visited[entry as usize] = true;
    // (dist, id) frontier — top-ef closest known.
    let mut top: Vec<(f32, u32)> = vec![(d0, entry)];
    // Candidates (min-heap emulated with sort).
    let mut cand: Vec<(f32, u32)> = vec![(d0, entry)];
    while let Some((cd, cid)) = pop_min(&mut cand) {
        let worst_top = top.last().map(|x| x.0).unwrap_or(f32::INFINITY);
        if cd > worst_top && top.len() >= ef { break; }
        for &nb in &adj[cid as usize] {
            if visited[nb as usize] { continue; }
            visited[nb as usize] = true;
            let nv = &vectors[nb as usize * DIM..(nb as usize + 1) * DIM];
            let d = l2_sq(nv, q);
            let worst = top.last().map(|x| x.0).unwrap_or(f32::INFINITY);
            if top.len() < ef || d < worst {
                push_sorted(&mut top, (d, nb), ef);
                cand.push((d, nb));
            }
        }
    }
    top
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
