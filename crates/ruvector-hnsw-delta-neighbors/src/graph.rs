//! Minimal k-NN graph used as a stand-in for an HNSW base-layer graph.
//!
//! HNSW's per-node structure at each layer is exactly a bounded adjacency
//! list. For the purposes of this crate — measuring how well neighbor lists
//! compress — a brute-force k-NN graph over random data is *harder* to
//! compress than an HNSW graph (HNSW has extra locality from the hierarchical
//! insertion), so it forms a conservative benchmark floor.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Bounded adjacency list. `adj[i]` is the neighbor list of node `i`.
#[derive(Debug, Clone)]
pub struct Graph {
    /// One neighbor list per node.
    pub adj: Vec<Vec<u32>>,
}

impl Graph {
    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.adj.len()
    }
    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.adj.is_empty()
    }
    /// Total edges across all lists.
    pub fn total_edges(&self) -> usize {
        self.adj.iter().map(|v| v.len()).sum()
    }
}

/// Generate `n` random unit-ish vectors in `d` dims with a fixed seed.
pub fn random_vectors(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..(n * d) {
        out.push(rng.gen_range(-1.0f32..1.0));
    }
    // Normalize each vector for stable cosine-like distances.
    for i in 0..n {
        let s = &mut out[i * d..(i + 1) * d];
        let norm: f32 = s.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in s.iter_mut() {
            *x /= norm;
        }
    }
    out
}

/// Squared Euclidean distance between two `d`-dim slices.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let t = a[i] - b[i];
        s += t * t;
    }
    s
}

/// Brute-force k-NN graph. O(n^2 * d); fine for the small benchmark corpora
/// this crate uses. Returns neighbor lists **sorted by ID** (not by distance)
/// so encodings can operate directly on ascending sequences.
pub fn brute_knn_graph(vectors: &[f32], n: usize, d: usize, k: usize) -> Graph {
    assert_eq!(vectors.len(), n * d);
    let mut adj = Vec::with_capacity(n);
    let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(n);
    for i in 0..n {
        scratch.clear();
        let vi = &vectors[i * d..(i + 1) * d];
        for j in 0..n {
            if j == i {
                continue;
            }
            let vj = &vectors[j * d..(j + 1) * d];
            scratch.push((sq_l2(vi, vj), j as u32));
        }
        let take = k.min(scratch.len());
        let nth = take.saturating_sub(1);
        scratch.select_nth_unstable_by(nth, |a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut nbrs: Vec<u32> = scratch[..take].iter().map(|x| x.1).collect();
        nbrs.sort_unstable();
        adj.push(nbrs);
    }
    Graph { adj }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brute_knn_produces_sorted_bounded_lists() {
        let v = random_vectors(64, 8, 42);
        let g = brute_knn_graph(&v, 64, 8, 8);
        assert_eq!(g.len(), 64);
        for row in &g.adj {
            assert_eq!(row.len(), 8);
            for w in row.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }
}
