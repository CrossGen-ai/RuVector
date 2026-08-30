//! Reordering strategies over an HNSW index's layer-0 adjacency.
//!
//! A [`Reordered`] index is a full copy of the input index whose `vectors`
//! buffer has been physically permuted, and whose neighbor lists have been
//! relabeled through the permutation, so that traversal walks contiguous
//! (or nearly contiguous) cache lines.

use crate::hnsw::HnswIndex;
use std::collections::VecDeque;

/// Produces a permutation `perm[new_id] = old_id`.
pub trait ReorderStrategy {
    fn name(&self) -> &'static str;
    fn permutation(&self, index: &HnswIndex) -> Vec<u32>;
}

/// Identity permutation (baseline: insertion order == physical order).
pub struct IdentityOrder;
impl ReorderStrategy for IdentityOrder {
    fn name(&self) -> &'static str { "identity" }
    fn permutation(&self, index: &HnswIndex) -> Vec<u32> {
        (0..index.len() as u32).collect()
    }
}

/// BFS from the HNSW entry point over the layer-0 adjacency. Nodes close in
/// graph-space become close in physical-space.
pub struct BfsOrder;
impl ReorderStrategy for BfsOrder {
    fn name(&self) -> &'static str { "bfs" }
    fn permutation(&self, index: &HnswIndex) -> Vec<u32> {
        let n = index.len();
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut q = VecDeque::new();
        if let Some(ep) = index.entry_point {
            visited[ep as usize] = true;
            q.push_back(ep);
        }
        while let Some(node) = q.pop_front() {
            order.push(node);
            for &nb in &index.layers[0][node as usize] {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    q.push_back(nb);
                }
            }
        }
        // Sweep any disconnected components in original id order.
        for i in 0..n as u32 {
            if !visited[i as usize] {
                visited[i as usize] = true;
                order.push(i);
                let mut q2 = VecDeque::from([i]);
                while let Some(node) = q2.pop_front() {
                    for &nb in &index.layers[0][node as usize] {
                        if !visited[nb as usize] {
                            visited[nb as usize] = true;
                            order.push(nb);
                            q2.push_back(nb);
                        }
                    }
                }
            }
        }
        order
    }
}

/// Reverse Cuthill-McKee ordering. Classic bandwidth reducer for sparse
/// symmetric matrices. Applied here to the symmetric closure of the HNSW L0
/// adjacency. Starts at a low-degree "pseudo-peripheral" node.
pub struct RcmOrder;
impl ReorderStrategy for RcmOrder {
    fn name(&self) -> &'static str { "rcm" }
    fn permutation(&self, index: &HnswIndex) -> Vec<u32> {
        let n = index.len();
        // Symmetrize adjacency.
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for u in 0..n {
            for &v in &index.layers[0][u] {
                adj[u].push(v);
                adj[v as usize].push(u as u32);
            }
        }
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
        }
        // Choose start: node with minimum degree.
        let mut start = 0usize;
        let mut best_deg = usize::MAX;
        for i in 0..n {
            if adj[i].len() < best_deg {
                best_deg = adj[i].len();
                start = i;
            }
        }
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        let mut q: VecDeque<u32> = VecDeque::new();
        visited[start] = true;
        q.push_back(start as u32);
        while let Some(node) = q.pop_front() {
            order.push(node);
            let mut nbs: Vec<u32> = adj[node as usize]
                .iter()
                .copied()
                .filter(|&x| !visited[x as usize])
                .collect();
            nbs.sort_by_key(|&x| adj[x as usize].len()); // ascending degree
            for nb in nbs {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    q.push_back(nb);
                }
            }
        }
        // Any disconnected components.
        for i in 0..n as u32 {
            if !visited[i as usize] {
                visited[i as usize] = true;
                order.push(i);
                let mut q2 = VecDeque::from([i]);
                while let Some(node) = q2.pop_front() {
                    let mut nbs: Vec<u32> = adj[node as usize]
                        .iter()
                        .copied()
                        .filter(|&x| !visited[x as usize])
                        .collect();
                    nbs.sort_by_key(|&x| adj[x as usize].len());
                    for nb in nbs {
                        if !visited[nb as usize] {
                            visited[nb as usize] = true;
                            order.push(nb);
                            q2.push_back(nb);
                        }
                    }
                }
            }
        }
        order.reverse(); // "reverse" in RCM
        order
    }
}

/// Physically reordered clone of an HNSW index. All ids are remapped so search
/// APIs continue to work; results are returned in the reordered id space and
/// can be mapped back through `new_to_old`.
#[derive(Debug, Clone)]
pub struct Reordered {
    pub index: HnswIndex,
    pub new_to_old: Vec<u32>,
    pub old_to_new: Vec<u32>,
}

impl Reordered {
    pub fn build<S: ReorderStrategy>(src: &HnswIndex, strat: &S) -> Self {
        let perm = strat.permutation(src);
        let n = src.len();
        assert_eq!(perm.len(), n, "permutation must cover all nodes");
        let mut old_to_new = vec![0u32; n];
        for (new_id, &old_id) in perm.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }
        // Rebuild vector buffer.
        let mut vectors = vec![0.0f32; n * src.dim];
        for (new_id, &old_id) in perm.iter().enumerate() {
            let src_slice = src.get(old_id);
            vectors[new_id * src.dim..(new_id + 1) * src.dim].copy_from_slice(src_slice);
        }
        // Rebuild layers with remapped ids.
        let mut layers: Vec<Vec<Vec<u32>>> = Vec::with_capacity(src.layers.len());
        for lvl in &src.layers {
            let mut new_lvl: Vec<Vec<u32>> = vec![Vec::new(); n];
            for (old_id, nbs) in lvl.iter().enumerate() {
                if old_id >= n {
                    break;
                }
                let new_id = old_to_new[old_id] as usize;
                new_lvl[new_id] = nbs.iter().map(|&x| old_to_new[x as usize]).collect();
            }
            layers.push(new_lvl);
        }
        let mut node_max_layer = vec![0u8; n];
        for (old_id, &ml) in src.node_max_layer.iter().enumerate() {
            node_max_layer[old_to_new[old_id] as usize] = ml;
        }
        let entry_point = src.entry_point.map(|ep| old_to_new[ep as usize]);
        let new_index = HnswIndex {
            cfg: src.cfg,
            dim: src.dim,
            vectors,
            layers,
            node_max_layer,
            entry_point,
            top_layer: src.top_layer,
            ..HnswIndex::new(src.dim, src.cfg)
        };
        Self { index: new_index, new_to_old: perm, old_to_new }
    }
}

/// Mean absolute id-gap along all layer-0 edges. A locality metric independent
/// of hardware: lower means neighbor lookups touch nearer physical addresses.
pub fn mean_edge_gap(index: &HnswIndex) -> f64 {
    let mut sum = 0i128;
    let mut count = 0i128;
    for (u, nbs) in index.layers[0].iter().enumerate() {
        for &v in nbs {
            let d = (u as i64 - v as i64).unsigned_abs() as i128;
            sum += d;
            count += 1;
        }
    }
    if count == 0 { 0.0 } else { sum as f64 / count as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::HnswConfig;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn build(n: usize, d: usize) -> HnswIndex {
        let mut idx = HnswIndex::new(d, HnswConfig::default());
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..n {
            let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>()).collect();
            idx.insert(&v);
        }
        idx
    }

    #[test]
    fn identity_preserves_search() {
        let src = build(200, 16);
        let re = Reordered::build(&src, &IdentityOrder);
        let q = vec![0.5f32; 16];
        let a = src.search_layer(&q, src.entry_point.unwrap(), 10, 0);
        let b = re.index.search_layer(&q, re.index.entry_point.unwrap(), 10, 0);
        // With identity perm, results (as ids) should be exactly equal.
        assert_eq!(a, b);
    }

    #[test]
    fn bfs_reduces_edge_gap() {
        let src = build(500, 16);
        let id_gap = mean_edge_gap(&src);
        let re = Reordered::build(&src, &BfsOrder);
        let bfs_gap = mean_edge_gap(&re.index);
        assert!(bfs_gap < id_gap, "bfs_gap {} !< identity_gap {}", bfs_gap, id_gap);
    }

    #[test]
    fn rcm_reduces_edge_gap() {
        let src = build(500, 16);
        let id_gap = mean_edge_gap(&src);
        let re = Reordered::build(&src, &RcmOrder);
        let rcm_gap = mean_edge_gap(&re.index);
        assert!(rcm_gap < id_gap, "rcm_gap {} !< identity_gap {}", rcm_gap, id_gap);
    }

    #[test]
    fn reordering_preserves_neighbor_set() {
        // The set of ids returned (mapped back through new_to_old) must equal
        // the set returned by the original index.
        let src = build(300, 12);
        let re = Reordered::build(&src, &RcmOrder);
        let q = vec![0.3f32; 12];
        let a: std::collections::HashSet<u32> = src
            .search_layer(&q, src.entry_point.unwrap(), 20, 0)
            .into_iter()
            .map(|(_, id)| id)
            .collect();
        let b: std::collections::HashSet<u32> = re
            .index
            .search_layer(&q, re.index.entry_point.unwrap(), 20, 0)
            .into_iter()
            .map(|(_, id)| re.new_to_old[id as usize])
            .collect();
        assert_eq!(a, b, "reorder must not change the search result set");
    }
}
