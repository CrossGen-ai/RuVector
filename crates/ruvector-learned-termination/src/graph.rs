//! Flat single-layer k-NN proximity graph — equivalent to HNSW layer-0.

use crate::dataset::l2sq;

#[derive(Clone, Debug)]
pub struct GraphConfig {
    pub k_neighbours: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig { k_neighbours: 16 }
    }
}

/// Single-layer k-NN proximity graph over f32 vectors.
///
/// Construction is O(n² · dim). PoC-appropriate for n <= ~10 k. The learned
/// termination logic is per-search-step and applies unchanged to a multi-layer
/// HNSW's layer-0 walk.
pub struct FlatGraph {
    pub vectors: Vec<Vec<f32>>,
    /// adjacency[i] = sorted list of (dist², neighbour_id).
    pub adjacency: Vec<Vec<(f32, usize)>>,
    pub config: GraphConfig,
}

impl FlatGraph {
    pub fn build(vectors: Vec<Vec<f32>>, config: GraphConfig) -> Self {
        let n = vectors.len();
        let k = config.k_neighbours.min(n.saturating_sub(1));
        let mut adjacency: Vec<Vec<(f32, usize)>> = vec![Vec::with_capacity(k); n];
        for i in 0..n {
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (l2sq(&vectors[i], &vectors[j]), j))
                .collect();
            dists.sort_unstable_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            dists.truncate(k);
            adjacency[i] = dists;
        }
        FlatGraph {
            vectors,
            adjacency,
            config,
        }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    pub fn dim(&self) -> usize {
        self.vectors.first().map(|v| v.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_unit_vectors;

    #[test]
    fn graph_has_k_neighbours() {
        let vecs = random_unit_vectors(30, 8, 7);
        let g = FlatGraph::build(vecs, GraphConfig { k_neighbours: 5 });
        for adj in &g.adjacency {
            assert_eq!(adj.len(), 5);
        }
    }

    #[test]
    fn graph_first_neighbour_is_closest() {
        let vecs = random_unit_vectors(30, 8, 7);
        let g = FlatGraph::build(vecs, GraphConfig { k_neighbours: 5 });
        for adj in &g.adjacency {
            for w in adj.windows(2) {
                assert!(w[0].0 <= w[1].0);
            }
        }
    }
}
