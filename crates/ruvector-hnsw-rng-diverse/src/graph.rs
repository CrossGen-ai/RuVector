//! Single-layer graph builder that plugs in any `Pruner`.
//!
//! For each point, take its top-`ef` nearest candidates by brute force, then
//! call `pruner.select(..)` to keep up to `m` out-edges. Undirected: also
//! back-links, re-pruning target nodes when they overflow.

use crate::{dist2, prune::Pruner};

pub struct GraphIndex {
    pub adj: Vec<Vec<usize>>,
    pub m: usize,
    pub name: &'static str,
    pub avg_degree: f32,
}

pub fn build_index(vectors: &[Vec<f32>], pruner: &dyn Pruner, m: usize, ef_build: usize) -> GraphIndex {
    let n = vectors.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::with_capacity(m); n];

    for i in 0..n {
        // Brute-force top-ef_build candidates (excluding self).
        let mut cands: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist2(&vectors[i], &vectors[j])))
            .collect();
        cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cands.truncate(ef_build);

        let kept = pruner.select(i, &cands, vectors, m);
        adj[i] = kept.clone();

        // Back-link: for each kept neighbor, add i and re-prune if needed.
        for nid in kept {
            adj[nid].push(i);
            if adj[nid].len() > m {
                // Recompute candidate list from current adjacency + re-prune.
                let mut bcands: Vec<(usize, f32)> = adj[nid]
                    .iter()
                    .map(|&j| (j, dist2(&vectors[nid], &vectors[j])))
                    .collect();
                bcands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                adj[nid] = pruner.select(nid, &bcands, vectors, m);
            }
        }
    }

    let total_deg: usize = adj.iter().map(|v| v.len()).sum();
    GraphIndex {
        adj,
        m,
        name: pruner.name(),
        avg_degree: total_deg as f32 / n as f32,
    }
}
