//! Node-relabelling strategies.
//!
//! Each strategy returns a permutation `perm[new_id] = old_id`.
//! `apply_permutation` rebuilds the CSR graph and vector store in the
//! new order, leaving search semantics identical.

use crate::graph::HnswGraph;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::VecDeque;

#[derive(Copy, Clone, Debug)]
pub enum Strategy {
    Identity,
    Bfs,
    Gorder { window: usize },
    Rgb { max_depth: usize },
}

pub fn identity_order(g: &HnswGraph) -> Vec<u32> {
    (0..g.n() as u32).collect()
}

/// Breadth-first from the entry point. Cheap; strong baseline.
pub fn bfs_order(g: &HnswGraph) -> Vec<u32> {
    let n = g.n();
    let mut perm = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    let mut q = VecDeque::new();
    q.push_back(g.entry);
    seen[g.entry as usize] = true;
    while let Some(v) = q.pop_front() {
        perm.push(v);
        for &nb in g.neighbours_of(v) {
            if !seen[nb as usize] {
                seen[nb as usize] = true;
                q.push_back(nb);
            }
        }
    }
    // Sweep any disconnected islands in original order.
    for i in 0..n as u32 {
        if !seen[i as usize] {
            perm.push(i);
        }
    }
    perm
}

/// Gorder (Wei & Karypis, KDD 2016), sliding-window variant.
///
/// Greedy: at each step, from the pool of unplaced vertices pick the one
/// with maximum gain = (# neighbours in the last `window` placed) +
/// (# 2-hop co-neighbours in the last window). Ties broken by id.
///
/// We use a bucketed frequency table over the last-window window instead
/// of the classic priority queue; this is O(n * avg_deg^2) but with a
/// small constant and matches Gorder's published quality on small graphs.
pub fn gorder(g: &HnswGraph, window: usize) -> Vec<u32> {
    let n = g.n();
    if n == 0 {
        return Vec::new();
    }
    let mut placed = vec![false; n];
    let mut perm: Vec<u32> = Vec::with_capacity(n);

    // Score contribution from each vertex currently in the sliding window.
    // score[u] += k when u is a neighbour (1-hop) or co-neighbour (2-hop) of a
    // recently placed vertex.
    let mut score = vec![0i32; n];
    let mut window_q: VecDeque<u32> = VecDeque::with_capacity(window + 1);

    // Seed: start at entry.
    let seed = g.entry as usize;
    place(&mut perm, &mut placed, seed as u32);
    update_scores(g, &mut score, &mut window_q, window, seed as u32, 1);

    while perm.len() < n {
        // Choose next: argmax score among unplaced. Ties -> smallest id.
        let mut best = -1i64;
        let mut best_id: u32 = u32::MAX;
        for i in 0..n {
            if placed[i] {
                continue;
            }
            let s = score[i] as i64;
            if s > best || (s == best && (i as u32) < best_id) {
                best = s;
                best_id = i as u32;
            }
        }
        if best_id == u32::MAX {
            // Isolated remainder; append in original order.
            for i in 0..n {
                if !placed[i] {
                    place(&mut perm, &mut placed, i as u32);
                }
            }
            break;
        }
        place(&mut perm, &mut placed, best_id);
        update_scores(g, &mut score, &mut window_q, window, best_id, 1);
    }
    perm
}

#[inline]
fn place(perm: &mut Vec<u32>, placed: &mut [bool], v: u32) {
    perm.push(v);
    placed[v as usize] = true;
}

fn update_scores(
    g: &HnswGraph,
    score: &mut [i32],
    window_q: &mut VecDeque<u32>,
    window: usize,
    just_placed: u32,
    sign: i32,
) {
    // Add contribution of just-placed vertex to score of its 1-hop + 2-hop.
    for &nb in g.neighbours_of(just_placed) {
        score[nb as usize] += sign;
        for &nb2 in g.neighbours_of(nb) {
            if nb2 != just_placed {
                score[nb2 as usize] += sign;
            }
        }
    }
    window_q.push_back(just_placed);
    if window_q.len() > window {
        let evicted = window_q.pop_front().unwrap();
        for &nb in g.neighbours_of(evicted) {
            score[nb as usize] -= sign;
            for &nb2 in g.neighbours_of(nb) {
                if nb2 != evicted {
                    score[nb2 as usize] -= sign;
                }
            }
        }
    }
}

/// Recursive graph bisection (Dhulipala et al. 2016 / Chierichetti et al. 2009).
///
/// Objective (log-gap cost): minimise sum over edges (u,v) of
///   log2(|pos(u) - pos(v)|). We approximate with a coordinate-descent
/// swap pass at each split: partition vertices into halves L | R,
/// compute per-vertex move gain, sort by gain, swap top-k pairs until no
/// improvement, then recurse.
///
/// This is the algorithm used to reorder web graphs, inverted indexes,
/// and recently graph-ANN indexes.
pub fn rgb_order(g: &HnswGraph, max_depth: usize) -> Vec<u32> {
    // Start from BFS order (a strong starting point) then apply RGB
    // top-down. Two coordinate-descent sweeps per split.
    let mut ids: Vec<u32> = bfs_order(g);
    let mut work = vec![0i32; g.n()];
    rgb_recurse(g, &mut ids, 0, max_depth, &mut work);
    ids
}

fn rgb_recurse(
    g: &HnswGraph,
    slice: &mut [u32],
    depth: usize,
    max_depth: usize,
    work: &mut [i32],
) {
    let n = slice.len();
    if n < 4 || depth >= max_depth {
        return;
    }
    let mid = n / 2;
    // Multiple coordinate-descent sweeps per split — helps escape a bad start.
    for _sweep in 0..3 {
    let _ = ();
    // Compute per-vertex "move gain": gain[i] = deg_R(i) - deg_L(i) where L,R are current halves.
    // Moving from L to R improves cost when gain > 0 (edges pull it right).
    // Build a lookup: side[node_id] = 0 if in L, 1 if in R.
    let mut side = vec![2u8; g.n()];
    for (i, &v) in slice.iter().enumerate() {
        side[v as usize] = if i < mid { 0 } else { 1 };
    }
    for i in 0..n {
        let v = slice[i];
        let (mut dl, mut dr) = (0i32, 0i32);
        for &nb in g.neighbours_of(v) {
            match side[nb as usize] {
                0 => dl += 1,
                1 => dr += 1,
                _ => {}
            }
        }
        // want to be on the majority-degree side.
        // gain toward L-membership = dl - dr; toward R = dr - dl.
        let s = side[v as usize];
        work[i] = if s == 0 { dr - dl } else { dl - dr };
    }
    // Sort L half by descending gain (want-to-move-right) and R half by descending gain (want-left).
    let mut left_idx: Vec<usize> = (0..mid).collect();
    let mut right_idx: Vec<usize> = (mid..n).collect();
    left_idx.sort_by(|&a, &b| work[b].cmp(&work[a]));
    right_idx.sort_by(|&a, &b| work[b].cmp(&work[a]));
    // Swap top pairs while both gains sum > 0.
    let pairs = left_idx.len().min(right_idx.len());
    let mut any = false;
    for k in 0..pairs {
        let li = left_idx[k];
        let ri = right_idx[k];
        if work[li] + work[ri] <= 0 {
            break;
        }
        slice.swap(li, ri);
        any = true;
    }
    if !any {
        break;
    }
    }
    // Recurse.
    let (l, r) = slice.split_at_mut(mid);
    rgb_recurse(g, l, depth + 1, max_depth, work);
    rgb_recurse(g, r, depth + 1, max_depth, work);
}

/// Rebuild a graph so that node with old id `perm[i]` sits at new id `i`.
pub fn apply_permutation(g: &HnswGraph, perm: &[u32]) -> HnswGraph {
    let n = g.n();
    assert_eq!(perm.len(), n);
    let mut inverse = vec![0u32; n];
    for (new_id, &old_id) in perm.iter().enumerate() {
        inverse[old_id as usize] = new_id as u32;
    }
    // Rebuild vectors.
    let mut new_data = vec![0.0f32; g.data.len()];
    for new_id in 0..n {
        let old_id = perm[new_id];
        let src = old_id as usize * g.dim;
        let dst = new_id * g.dim;
        new_data[dst..dst + g.dim].copy_from_slice(&g.data[src..src + g.dim]);
    }
    // Rebuild adjacency (relabel + sort by new id for prefetch friendliness).
    let mut new_offsets = Vec::with_capacity(n + 1);
    new_offsets.push(0u32);
    let mut new_neighbours: Vec<u32> = Vec::with_capacity(g.neighbours.len());
    for new_id in 0..n {
        let old_id = perm[new_id];
        let nbs = g.neighbours_of(old_id);
        let mut mapped: Vec<u32> = nbs.iter().map(|&x| inverse[x as usize]).collect();
        mapped.sort_unstable();
        new_neighbours.extend_from_slice(&mapped);
        new_offsets.push(new_neighbours.len() as u32);
    }
    HnswGraph {
        dim: g.dim,
        data: new_data,
        offsets: new_offsets,
        neighbours: new_neighbours,
        entry: inverse[g.entry as usize],
    }
}

/// Log-gap cost proxy: mean log2 of |neighbour_id - self_id|.
/// Lower is better; correlates with cache-friendliness in graph traversal.
pub fn log_gap_cost(g: &HnswGraph) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for i in 0..g.n() {
        for &nb in g.neighbours_of(i as u32) {
            let d = (nb as i64 - i as i64).unsigned_abs().max(1) as f64;
            sum += d.log2();
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}
