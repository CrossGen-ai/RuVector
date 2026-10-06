//! Instrumented greedy beam search with pluggable entry strategy.

use rand::Rng;

use crate::graph::Nsw;
use crate::lsh::LshSeedIndex;
use crate::metrics::{SearchStats, l2_sq};

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub ef: usize,
    pub k: usize,
}

/// The thing we're comparing. All three strategies share the same beam
/// walk and the same termination rule — only the first node differs.
#[derive(Clone)]
pub enum EntryStrategy<'a> {
    Fixed,
    Random,
    Lsh(&'a LshSeedIndex),
}

pub fn search<R: Rng>(
    index: &Nsw,
    query: &[f32],
    params: SearchParams,
    strategy: &EntryStrategy,
    rng: &mut R,
) -> (Vec<u32>, SearchStats) {
    let entry = match strategy {
        EntryStrategy::Fixed => index.fixed_entry,
        EntryStrategy::Random => rng.gen_range(0..index.n as u32),
        EntryStrategy::Lsh(idx) => idx.seed_for(query),
    };

    let mut stats = SearchStats::default();
    let d_entry = l2_sq(index.vector(entry), query);
    stats.distance_computations += 1;
    stats.entry_distance = d_entry;

    let mut visited = vec![false; index.n];
    visited[entry as usize] = true;

    let mut results: Vec<(f32, u32)> = Vec::with_capacity(params.ef);
    results.push((d_entry, entry));

    let mut frontier: Vec<(f32, u32)> = vec![(d_entry, entry)];
    while let Some((d_c, c)) = pop_min(&mut frontier) {
        let worst =
            results.last().copied().map(|p| p.0).unwrap_or(f32::INFINITY);
        if d_c > worst && results.len() >= params.ef {
            break;
        }
        stats.hops += 1;
        for &nb in index.neighbors_of(c) {
            if nb == u32::MAX {
                continue;
            }
            if visited[nb as usize] {
                continue;
            }
            visited[nb as usize] = true;
            let d_nb = l2_sq(index.vector(nb), query);
            stats.distance_computations += 1;
            let worst_now =
                results.last().copied().map(|p| p.0).unwrap_or(f32::INFINITY);
            if results.len() < params.ef || d_nb < worst_now {
                insert_sorted(&mut results, (d_nb, nb));
                if results.len() > params.ef {
                    results.pop();
                }
                frontier.push((d_nb, nb));
            }
        }
    }

    let topk: Vec<u32> =
        results.iter().take(params.k).map(|p| p.1).collect();
    (topk, stats)
}

fn pop_min(v: &mut Vec<(f32, u32)>) -> Option<(f32, u32)> {
    if v.is_empty() {
        return None;
    }
    let mut mi = 0;
    for i in 1..v.len() {
        if v[i].0 < v[mi].0 {
            mi = i;
        }
    }
    Some(v.swap_remove(mi))
}

fn insert_sorted(v: &mut Vec<(f32, u32)>, item: (f32, u32)) {
    let pos = v.iter().position(|p| p.0 > item.0).unwrap_or(v.len());
    v.insert(pos, item);
}

/// Brute-force top-k used to score recall. Not in the hot path.
pub fn brute_force_topk(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = vectors.len() / dim;
    let mut all: Vec<(f32, u32)> = (0..n as u32)
        .map(|i| (l2_sq(&vectors[i as usize * dim..(i as usize + 1) * dim], query), i))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    all.into_iter().take(k).map(|p| p.1).collect()
}
