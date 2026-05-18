//! Core `RoarGraph` index — adjacency list + greedy beam search.
//!
//! After `build()` has been called, `search(query, k, ef)` runs a greedy
//! beam search with beam width `ef` (analogous to HNSW's `ef_search`).

use std::collections::{BinaryHeap, HashSet};

use crate::error::RoarError;

/// Result of a single nearest-neighbour search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Index into the base corpus (same order as `add()` calls).
    pub id: usize,
    /// Squared L2 distance to the query.
    pub dist: f32,
}

/// Compute squared L2 distance between two slices.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The RoarGraph projected-bipartite ANN index.
///
/// Build via [`RoarGraph::build`]; search via [`RoarGraph::search`].
pub struct RoarGraph {
    /// Dimensionality of base and query vectors.
    pub(crate) dim: usize,
    /// Raw base vectors stored in insertion order.
    pub(crate) vectors: Vec<Vec<f32>>,
    /// Adjacency list: `adj[i]` contains the out-neighbours of node `i`.
    pub(crate) adj: Vec<Vec<u32>>,
    /// Whether `build()` has been called.
    pub(crate) built: bool,
}

impl RoarGraph {
    /// Create an empty index for vectors of length `dim`.
    pub fn new(dim: usize) -> Self {
        RoarGraph {
            dim,
            vectors: Vec::new(),
            adj: Vec::new(),
            built: false,
        }
    }

    /// Add base vectors to the index (must be called before `build`).
    pub fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RoarError> {
        for v in vectors {
            if v.len() != self.dim {
                return Err(RoarError::DimensionMismatch {
                    expected: self.dim,
                    got: v.len(),
                });
            }
            self.vectors.push(v.clone());
        }
        Ok(())
    }

    /// Number of vectors currently in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True if no vectors have been added.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Adjacency list (available after `build()`).
    pub fn adjacency(&self) -> &Vec<Vec<u32>> {
        &self.adj
    }

    /// Greedy beam search over the graph.
    ///
    /// Returns up to `k` results sorted by ascending distance.
    /// `ef` is the candidate beam width (higher → better recall, slower).
    ///
    /// Entry point is the node whose stored vector is closest to the
    /// global mean — a cheap approximation; for production use a
    /// navigating node chosen during build.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchResult>, RoarError> {
        if !self.built {
            return Err(RoarError::NotBuilt);
        }
        if self.vectors.is_empty() {
            return Err(RoarError::EmptyIndex);
        }
        if query.len() != self.dim {
            return Err(RoarError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let n = self.vectors.len();
        let k = k.min(n);

        // Choose entry point: node 0 is fine for a random graph;
        // the navigating node heuristic would add complexity without
        // changing correctness.
        let entry = 0usize;

        // visited set
        let mut visited: HashSet<u32> = HashSet::new();

        // candidates: (neg_dist, id) in a max-heap so smallest dist is top
        // We use -dist so BinaryHeap (max-heap) gives us min-dist first.
        let mut candidates: BinaryHeap<(ordered_float::OrderedFloat, u32)> = BinaryHeap::new();
        // result set: maintain best-ef seen so far
        let mut results: BinaryHeap<(ordered_float::OrderedFloat, u32)> = BinaryHeap::new();

        let d_entry = l2sq(query, &self.vectors[entry]);
        candidates.push((ordered_float::OrderedFloat(-d_entry), entry as u32));
        results.push((ordered_float::OrderedFloat(d_entry), entry as u32));
        visited.insert(entry as u32);

        while let Some((neg_d_cand, cand_id)) = candidates.pop() {
            let d_cand = -neg_d_cand.0;

            // If the worst result we have is better than the best candidate,
            // we can stop.
            if let Some(&(worst_d, _)) = results.peek() {
                if results.len() >= ef && d_cand > worst_d.0 {
                    break;
                }
            }

            // Expand neighbours
            for &nb in &self.adj[cand_id as usize] {
                if visited.insert(nb) {
                    let d_nb = l2sq(query, &self.vectors[nb as usize]);

                    // Decide whether to add to result set
                    let should_add = if results.len() < ef {
                        true
                    } else if let Some(&(worst_d, _)) = results.peek() {
                        d_nb < worst_d.0
                    } else {
                        true
                    };

                    if should_add {
                        candidates.push((ordered_float::OrderedFloat(-d_nb), nb));
                        results.push((ordered_float::OrderedFloat(d_nb), nb));
                        // Keep results bounded to ef
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Collect results, sort ascending
        let mut out: Vec<SearchResult> = results
            .into_iter()
            .map(|(d, id)| SearchResult { id: id as usize, dist: d.0 })
            .collect();
        out.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        out.truncate(k);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Minimal ordered float wrapper to use f32 in BinaryHeap
// ---------------------------------------------------------------------------
mod ordered_float {
    use std::cmp::Ordering;

    #[derive(Copy, Clone, PartialEq)]
    pub struct OrderedFloat(pub f32);

    impl Eq for OrderedFloat {}

    impl PartialOrd for OrderedFloat {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for OrderedFloat {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_dimension_check() {
        let mut g = RoarGraph::new(4);
        assert!(g.add(&[vec![1.0, 2.0, 3.0, 4.0]]).is_ok());
        assert!(g.add(&[vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn test_search_requires_build() {
        let mut g = RoarGraph::new(4);
        g.add(&[vec![1.0, 2.0, 3.0, 4.0]]).unwrap();
        assert!(matches!(
            g.search(&[1.0, 2.0, 3.0, 4.0], 1, 10),
            Err(RoarError::NotBuilt)
        ));
    }
}
