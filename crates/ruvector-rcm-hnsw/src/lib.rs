//! RCM-HNSW: Reverse Cuthill-McKee reordering for HNSW layer-0 adjacency.
//!
//! The idea: after an HNSW index is built, treat its layer-0 adjacency as a
//! symmetric sparse graph. Apply Reverse Cuthill-McKee (RCM) node ordering
//! to minimize bandwidth. Vectors and adjacency lists are then permuted
//! so that neighbors of a node sit near it in memory. The effect is fewer
//! TLB/L1 misses during BFS-style traversal.
//!
//! This crate contains three orderings:
//!   * Identity — baseline, insertion order.
//!   * Random   — worst case, Fisher-Yates permutation.
//!   * RCM      — Cuthill-McKee BFS from a minimum-degree root, reversed.

use rand::prelude::*;
use std::collections::VecDeque;

pub type NodeId = u32;
pub type Vector = Vec<f32>;

#[derive(Clone)]
pub struct HnswLayer0 {
    pub vectors: Vec<Vector>,
    pub adjacency: Vec<Vec<NodeId>>,
    pub dim: usize,
}

impl HnswLayer0 {
    pub fn n(&self) -> usize {
        self.vectors.len()
    }

    /// Permute in place: `perm[new_id] = old_id`.
    pub fn permute(&self, perm: &[NodeId]) -> HnswLayer0 {
        let n = self.n();
        assert_eq!(perm.len(), n);
        let mut inv = vec![0u32; n];
        for (new_id, &old_id) in perm.iter().enumerate() {
            inv[old_id as usize] = new_id as u32;
        }
        let mut vectors = Vec::with_capacity(n);
        let mut adjacency = Vec::with_capacity(n);
        for &old in perm {
            vectors.push(self.vectors[old as usize].clone());
            let remapped: Vec<NodeId> = self.adjacency[old as usize]
                .iter()
                .map(|&nb| inv[nb as usize])
                .collect();
            adjacency.push(remapped);
        }
        HnswLayer0 { vectors, adjacency, dim: self.dim }
    }

    /// Mean |old_id - neighbor_id| gap in raw node ids — proxy for locality.
    pub fn mean_edge_gap(&self) -> f64 {
        let mut total: u64 = 0;
        let mut edges: u64 = 0;
        for (i, nbs) in self.adjacency.iter().enumerate() {
            for &nb in nbs {
                let d = (i as i64 - nb as i64).unsigned_abs();
                total += d;
                edges += 1;
            }
        }
        if edges == 0 { 0.0 } else { total as f64 / edges as f64 }
    }
}

/// Identity permutation.
pub fn perm_identity(n: usize) -> Vec<NodeId> {
    (0..n as u32).collect()
}

/// Random permutation.
pub fn perm_random(n: usize, seed: u64) -> Vec<NodeId> {
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
    let mut v: Vec<NodeId> = (0..n as u32).collect();
    v.shuffle(&mut rng);
    v
}

/// Reverse Cuthill-McKee: BFS from a min-degree root, neighbors visited
/// in increasing-degree order, then the resulting order is reversed.
/// Handles disconnected graphs by restarting from the next unvisited
/// min-degree node.
pub fn perm_rcm(adjacency: &[Vec<NodeId>]) -> Vec<NodeId> {
    let n = adjacency.len();
    let degree: Vec<usize> = adjacency.iter().map(|a| a.len()).collect();
    let mut visited = vec![false; n];
    let mut order: Vec<NodeId> = Vec::with_capacity(n);

    // Nodes sorted by degree, used as the pool of roots for each component.
    let mut by_deg: Vec<usize> = (0..n).collect();
    by_deg.sort_by_key(|&i| degree[i]);
    let mut root_cursor = 0;

    while order.len() < n {
        // Find next unvisited root.
        while root_cursor < n && visited[by_deg[root_cursor]] {
            root_cursor += 1;
        }
        if root_cursor >= n {
            break;
        }
        let root = by_deg[root_cursor] as NodeId;
        let mut q: VecDeque<NodeId> = VecDeque::new();
        q.push_back(root);
        visited[root as usize] = true;

        while let Some(u) = q.pop_front() {
            order.push(u);
            let mut nbs: Vec<NodeId> = adjacency[u as usize]
                .iter()
                .copied()
                .filter(|&v| !visited[v as usize])
                .collect();
            nbs.sort_unstable();
            nbs.dedup();
            nbs.sort_by_key(|&v| degree[v as usize]);
            for v in nbs {
                if visited[v as usize] { continue; }
                visited[v as usize] = true;
                q.push_back(v);
            }
        }
    }
    order.reverse();
    order
}

/// Euclidean distance squared.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Greedy best-first search on a layer-0 graph, returning top-k candidates
/// (ascending distance). `ef` bounds the dynamic candidate list.
/// Returns (top_ids, visited_count).
pub fn search(
    graph: &HnswLayer0,
    query: &[f32],
    entry: NodeId,
    k: usize,
    ef: usize,
) -> (Vec<(NodeId, f32)>, usize) {
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    #[derive(PartialEq)]
    struct Item(f32, NodeId);
    impl Eq for Item {}
    impl Ord for Item {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
        }
    }
    impl PartialOrd for Item {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) }
    }

    let n = graph.n();
    let mut visited = vec![false; n];
    let mut candidates: BinaryHeap<Reverse<Item>> = BinaryHeap::new(); // min-heap by dist
    let mut top: BinaryHeap<Item> = BinaryHeap::new(); // max-heap by dist, size ef
    let mut visited_count = 0usize;

    let d0 = l2sq(&graph.vectors[entry as usize], query);
    candidates.push(Reverse(Item(d0, entry)));
    top.push(Item(d0, entry));
    visited[entry as usize] = true;
    visited_count += 1;

    while let Some(Reverse(Item(cd, cu))) = candidates.pop() {
        let furthest = top.peek().map(|x| x.0).unwrap_or(f32::MAX);
        if cd > furthest && top.len() >= ef {
            break;
        }
        for &nb in &graph.adjacency[cu as usize] {
            if visited[nb as usize] { continue; }
            visited[nb as usize] = true;
            visited_count += 1;
            let dv = l2sq(&graph.vectors[nb as usize], query);
            let worst = top.peek().map(|x| x.0).unwrap_or(f32::MAX);
            if top.len() < ef || dv < worst {
                candidates.push(Reverse(Item(dv, nb)));
                top.push(Item(dv, nb));
                if top.len() > ef { top.pop(); }
            }
        }
    }

    let mut out: Vec<(NodeId, f32)> = top.into_iter().map(|Item(d, i)| (i, d)).collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    out.truncate(k);
    (out, visited_count)
}

/// Build a toy HNSW-like layer-0 by connecting each node to its `m` approximate
/// nearest neighbors found through a small-world sample. Not a real HNSW
/// builder, but it produces a graph with the same traversal shape so the
/// reordering benchmark is honest.
pub fn build_smallworld_layer(
    vectors: Vec<Vector>,
    m: usize,
    sample: usize,
    seed: u64,
) -> HnswLayer0 {
    let n = vectors.len();
    let dim = vectors[0].len();
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
    let mut adjacency: Vec<Vec<NodeId>> = vec![Vec::with_capacity(m); n];

    for i in 0..n {
        let mut cands: Vec<(f32, NodeId)> = Vec::with_capacity(sample);
        for _ in 0..sample {
            let j = rng.gen_range(0..n);
            if j == i { continue; }
            let d = l2sq(&vectors[i], &vectors[j]);
            cands.push((d, j as NodeId));
        }
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cands.truncate(m);
        adjacency[i] = cands.into_iter().map(|(_, j)| j).collect();
    }

    // Symmetrize: ensure undirected edges so RCM behaves.
    let mut extra: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for (i, nbs) in adjacency.iter().enumerate() {
        for &j in nbs {
            if !adjacency[j as usize].contains(&(i as NodeId)) {
                extra[j as usize].push(i as NodeId);
            }
        }
    }
    for i in 0..n {
        adjacency[i].extend(extra[i].drain(..));
        adjacency[i].sort_unstable();
        adjacency[i].dedup();
        if let Some(pos) = adjacency[i].iter().position(|&v| v == i as NodeId) {
            adjacency[i].swap_remove(pos);
        }
    }

    HnswLayer0 { vectors, adjacency, dim }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_graph() -> HnswLayer0 {
        let vectors: Vec<Vector> = (0..8)
            .map(|i| vec![i as f32, (i * 2) as f32])
            .collect();
        // Ring + chords
        let adjacency: Vec<Vec<NodeId>> = vec![
            vec![1, 7],
            vec![0, 2],
            vec![1, 3, 5],
            vec![2, 4],
            vec![3, 5],
            vec![2, 4, 6],
            vec![5, 7],
            vec![6, 0],
        ];
        HnswLayer0 { vectors, adjacency, dim: 2 }
    }

    #[test]
    fn rcm_is_permutation() {
        let g = tiny_graph();
        let p = perm_rcm(&g.adjacency);
        assert_eq!(p.len(), g.n());
        let mut seen = vec![false; g.n()];
        for &id in &p {
            assert!(!seen[id as usize]);
            seen[id as usize] = true;
        }
    }

    #[test]
    fn rcm_reduces_mean_edge_gap_on_ring() {
        // Pure ring: identity ordering is already perfect; any RCM should
        // at worst match it.
        let n = 32;
        let vectors: Vec<Vector> = (0..n).map(|i| vec![i as f32]).collect();
        let adjacency: Vec<Vec<NodeId>> = (0..n as NodeId)
            .map(|i| vec![((i + n as NodeId - 1) % n as NodeId), (i + 1) % n as NodeId])
            .collect();
        let g = HnswLayer0 { vectors, adjacency, dim: 1 };
        let base = g.mean_edge_gap();
        let p = perm_rcm(&g.adjacency);
        let g2 = g.permute(&p);
        let after = g2.mean_edge_gap();
        assert!(after <= base + 0.5, "RCM should not worsen a ring (base={}, after={})", base, after);
    }

    #[test]
    fn rcm_helps_random_graph() {
        let vectors: Vec<Vector> = (0..512)
            .map(|i| {
                let a = (i as f32).sin();
                let b = (i as f32).cos();
                vec![a, b, a * b]
            })
            .collect();
        let g = build_smallworld_layer(vectors, 8, 24, 42);
        let rand_g = g.permute(&perm_random(g.n(), 7));
        let rcm_g = g.permute(&perm_rcm(&g.adjacency));
        let base = rand_g.mean_edge_gap();
        let after = rcm_g.mean_edge_gap();
        assert!(after < base, "RCM should beat random (base={}, after={})", base, after);
    }

    #[test]
    fn search_finds_same_set_regardless_of_order() {
        let vectors: Vec<Vector> = (0..256)
            .map(|i| vec![(i as f32).sin(), (i as f32).cos(), (i as f32 * 0.3).sin()])
            .collect();
        let g = build_smallworld_layer(vectors, 8, 32, 11);
        let q = vec![0.5_f32, -0.5, 0.1];

        let perm = perm_rcm(&g.adjacency);
        let g2 = g.permute(&perm);
        let inv: Vec<u32> = {
            let mut iv = vec![0u32; g.n()];
            for (new_id, &old) in perm.iter().enumerate() { iv[old as usize] = new_id as u32; }
            iv
        };

        let (res_orig, _) = search(&g, &q, 0, 10, 32);
        let (res_perm, _) = search(&g2, &q, inv[0], 10, 32);
        let map_perm: Vec<u32> = res_perm.iter().map(|(id, _)| perm[*id as usize]).collect();
        let set_orig: std::collections::HashSet<u32> = res_orig.iter().map(|(id, _)| *id).collect();
        let set_perm: std::collections::HashSet<u32> = map_perm.into_iter().collect();
        let overlap = set_orig.intersection(&set_perm).count();
        assert!(overlap >= 8, "top-10 should overlap >=8 after permutation, got {}", overlap);
    }
}
