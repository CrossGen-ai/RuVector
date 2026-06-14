//! Minimum spanning tree over a (small) leaf bucket.
//!
//! HCNNG uses Prim's algorithm because the leaf bucket size is bounded by
//! `leaf_size` (typically 16–64), so an O(L^2) implementation with no heap
//! beats Kruskal+union-find in practice on small dense inputs.
//!
//! Returns the set of MST edges as (u, v) pairs of global dataset indices.
//!
//! We also expose `mst_plus_knn_edges`: HCNNG's MST plus the k nearest
//! neighbors of each leaf member. The kNN edges densify the local graph
//! and dramatically improve recall on multi-cluster data — the bare MST
//! is path-shaped and the beam search can stall on it.

use crate::distance::Distance;

pub fn mst_edges<D: Distance + ?Sized>(
    leaf: &[u32],
    vectors: &[Vec<f32>],
    dist: &D,
) -> Vec<(u32, u32)> {
    let n = leaf.len();
    if n < 2 {
        return Vec::new();
    }

    // best_w[i] = current minimum distance from leaf[i] to the growing MST.
    // best_from[i] = which vertex in the MST it would connect to.
    let mut best_w = vec![f32::INFINITY; n];
    let mut best_from = vec![u32::MAX; n];
    let mut in_tree = vec![false; n];

    // Start from local index 0.
    in_tree[0] = true;
    let v0 = &vectors[leaf[0] as usize];
    for j in 1..n {
        let w = dist.d(v0, &vectors[leaf[j] as usize]);
        best_w[j] = w;
        best_from[j] = leaf[0];
    }

    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(n - 1);
    for _ in 1..n {
        // Pick the non-tree vertex with smallest best_w.
        let mut best = usize::MAX;
        let mut best_v = f32::INFINITY;
        for j in 0..n {
            if !in_tree[j] && best_w[j] < best_v {
                best_v = best_w[j];
                best = j;
            }
        }
        if best == usize::MAX {
            break;
        }
        in_tree[best] = true;
        edges.push((best_from[best], leaf[best]));

        // Relax remaining vertices.
        let vb = &vectors[leaf[best] as usize];
        for j in 0..n {
            if !in_tree[j] {
                let w = dist.d(vb, &vectors[leaf[j] as usize]);
                if w < best_w[j] {
                    best_w[j] = w;
                    best_from[j] = leaf[best];
                }
            }
        }
    }
    edges
}

/// MST edges PLUS the `knn_per_node` nearest neighbors of each leaf member
/// (within the leaf). Pure O(L^2) using one distance matrix pass.
pub fn mst_plus_knn_edges<D: Distance + ?Sized>(
    leaf: &[u32],
    vectors: &[Vec<f32>],
    dist: &D,
    knn_per_node: usize,
) -> Vec<(u32, u32)> {
    let n = leaf.len();
    let mut edges = mst_edges(leaf, vectors, dist);
    if knn_per_node == 0 || n < 2 {
        return edges;
    }
    // Compute full L x L distance matrix once.
    let mut dm = vec![0.0f32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let w = dist.d(&vectors[leaf[i] as usize], &vectors[leaf[j] as usize]);
            dm[i * n + j] = w;
            dm[j * n + i] = w;
        }
    }
    for i in 0..n {
        let mut row: Vec<(f32, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (dm[i * n + j], j))
            .collect();
        row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, j) in row.into_iter().take(knn_per_node) {
            edges.push((leaf[i], leaf[j]));
        }
    }
    edges
}
