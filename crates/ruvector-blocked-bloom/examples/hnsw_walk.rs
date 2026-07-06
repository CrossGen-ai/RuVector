//! Minimal HNSW-shaped traversal using the [`VisitedSet`] trait.
//!
//! Not a full HNSW — just a demonstration that all three impls plug into the
//! same beam-search skeleton and produce identical (or ε-close, for Bloom)
//! visit counts.

use rand::{Rng, SeedableRng, rngs::StdRng};
use ruvector_blocked_bloom::{
    BitmapVisited, BlockedBloomVisited, HashSetVisited, VisitedSet,
};

/// Fake graph: each node's neighbors are deterministic from its id, seeded.
fn neighbors(id: u32, seed: u64, degree: usize) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed ^ id as u64);
    (0..degree).map(|_| rng.gen_range(0..1_000_000)).collect()
}

/// One beam-search "hop": expand every frontier node, dedup via `visited`.
fn expand<V: VisitedSet>(frontier: &[u32], visited: &mut V, degree: usize, seed: u64) -> Vec<u32> {
    let mut next = Vec::with_capacity(frontier.len() * degree);
    for &id in frontier {
        for n in neighbors(id, seed, degree) {
            if visited.mark(n) {
                next.push(n);
            }
        }
    }
    next
}

fn walk<V: VisitedSet>(mut visited: V, start: u32, hops: usize) -> (usize, &'static str) {
    let name = visited.name();
    visited.mark(start);
    let mut frontier = vec![start];
    let mut total_new = 1;
    for _ in 0..hops {
        frontier = expand(&frontier, &mut visited, 32, 0xABCD);
        total_new += frontier.len();
        // Cap the beam to keep this a bounded search.
        if frontier.len() > 128 {
            frontier.truncate(128);
        }
    }
    (total_new, name)
}

fn main() {
    let start: u32 = 42;
    let hops = 6;
    for (impl_kind, count) in [
        walk(HashSetVisited::with_capacity(1024), start, hops),
        walk(BitmapVisited::new(1_000_000), start, hops),
        walk(
            BlockedBloomVisited::for_load(4096),
            start,
            hops,
        ),
    ]
    .into_iter()
    .map(|(count, name)| (name, count))
    {
        println!("{impl_kind:<28} visited {count} unique nodes");
    }
}
