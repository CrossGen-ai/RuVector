//! Elias-Fano compressed HNSW adjacency lists.
//!
//! HNSW neighbor lists are usually stored as `Vec<u32>` per node (32 bits per
//! neighbor id). When ids are drawn from `[0, N)`, Elias-Fano encoding of the
//! sorted neighbor set uses roughly `2 + log2(N / M)` bits per neighbor while
//! supporting random `select(i)` and sequential scans suitable for graph
//! traversal.
//!
//! This crate exposes three concrete adjacency backends behind an `Adjacency`
//! trait and provides a reproducible benchmark harness.

pub mod ef;
pub mod packed;
pub mod plain;

use std::fmt::Debug;

/// Backend-agnostic HNSW adjacency store.
///
/// All implementations expose the same iteration API: for a given node id,
/// yield the neighbor ids in some order. Elias-Fano iterates in sorted order,
/// which is also cache-friendly: the ids come out monotonically increasing.
pub trait Adjacency: Debug + Send + Sync {
    /// Number of nodes.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append neighbors for the next node. Neighbors are consumed as a slice;
    /// the backend may sort/compress internally.
    fn push(&mut self, neighbors: &[u32]);

    /// Total heap bytes used by the backend's internal buffers.
    fn bytes(&self) -> usize;

    /// Decode neighbors of `node` into `out` (clears `out` first).
    fn decode_into(&self, node: usize, out: &mut Vec<u32>);
}

/// A tiny deterministic PRNG for reproducible benchmarks (xorshift64*).
#[derive(Clone, Copy)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        Self { state: s }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    #[inline]
    pub fn gen_range(&mut self, n: u32) -> u32 {
        // Lemire's unbiased bounded range (good enough for benchmarks).
        let m = (self.next_u32() as u64).wrapping_mul(n as u64);
        (m >> 32) as u32
    }
}

/// Build a synthetic HNSW-like adjacency graph: `n` nodes, each with `m`
/// distinct neighbor ids drawn uniformly from `[0, n)`. Neighbor lists are
/// returned *sorted*, which matches what Elias-Fano consumes directly and
/// also matches a cache-friendly storage layout for the plain baseline.
pub fn build_random_graph(n: usize, m: usize, seed: u64) -> Vec<Vec<u32>> {
    assert!(m < n, "m must be < n");
    let mut rng = XorShift64::new(seed);
    let mut graph = Vec::with_capacity(n);
    let mut buf = Vec::with_capacity(m);
    for _ in 0..n {
        buf.clear();
        while buf.len() < m {
            let candidate = rng.gen_range(n as u32);
            if !buf.contains(&candidate) {
                buf.push(candidate);
            }
        }
        buf.sort_unstable();
        graph.push(buf.clone());
    }
    graph
}

/// Fill an `Adjacency` from a list-of-lists.
pub fn fill_from<A: Adjacency>(adj: &mut A, lists: &[Vec<u32>]) {
    for row in lists {
        adj.push(row);
    }
}

/// Tiny BFS-style traversal that mimics the inner loop of an HNSW search:
/// starting from `entry`, visit `budget` nodes in order of discovery,
/// decoding each neighbor list once. Returns the number of edges touched —
/// this is both a correctness anchor (same number across backends) and the
/// quantity we time in benchmarks.
pub fn bfs_touch<A: Adjacency>(adj: &A, entry: u32, budget: usize) -> u64 {
    let n = adj.len();
    let mut visited = vec![false; n];
    let mut queue: Vec<u32> = Vec::with_capacity(budget);
    let mut neighbors: Vec<u32> = Vec::with_capacity(64);
    queue.push(entry);
    visited[entry as usize] = true;
    let mut edges: u64 = 0;
    let mut head = 0usize;
    while head < queue.len() && head < budget {
        let node = queue[head] as usize;
        head += 1;
        adj.decode_into(node, &mut neighbors);
        edges += neighbors.len() as u64;
        for &nb in &neighbors {
            let idx = nb as usize;
            if !visited[idx] {
                visited[idx] = true;
                queue.push(nb);
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_backend<A: Adjacency>(adj: &mut A, lists: &[Vec<u32>]) {
        fill_from(adj, lists);
        assert_eq!(adj.len(), lists.len());
        let mut out = Vec::new();
        for (i, row) in lists.iter().enumerate() {
            adj.decode_into(i, &mut out);
            let mut sorted = row.clone();
            sorted.sort_unstable();
            assert_eq!(out, sorted, "row {} mismatch", i);
        }
    }

    #[test]
    fn roundtrip_all_backends_small() {
        let g = build_random_graph(256, 16, 42);
        roundtrip_backend(&mut plain::PlainAdjacency::new(), &g);
        roundtrip_backend(&mut packed::PackedAdjacency::new(256), &g);
        roundtrip_backend(&mut ef::EFAdjacency::new(256), &g);
    }

    #[test]
    fn roundtrip_all_backends_mid() {
        let g = build_random_graph(2048, 32, 7);
        roundtrip_backend(&mut plain::PlainAdjacency::new(), &g);
        roundtrip_backend(&mut packed::PackedAdjacency::new(2048), &g);
        roundtrip_backend(&mut ef::EFAdjacency::new(2048), &g);
    }

    #[test]
    fn bfs_edge_counts_match_across_backends() {
        let g = build_random_graph(1024, 24, 99);
        let mut plain = plain::PlainAdjacency::new();
        let mut packed = packed::PackedAdjacency::new(1024);
        let mut ef = ef::EFAdjacency::new(1024);
        fill_from(&mut plain, &g);
        fill_from(&mut packed, &g);
        fill_from(&mut ef, &g);
        let a = bfs_touch(&plain, 0, 200);
        let b = bfs_touch(&packed, 0, 200);
        let c = bfs_touch(&ef, 0, 200);
        assert_eq!(a, b);
        assert_eq!(b, c);
    }
}
