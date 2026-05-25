//! Segment-tree of NSW graphs — the practical SeRF approach.
//!
//! 1. Items are sorted by key, giving a rank in `[0, n)`.
//! 2. A segment tree is built over rank-space.
//! 3. Every internal node whose span is `>= leaf_size` owns an NSW built over
//!    the items in its span.
//! 4. A query range on keys is translated to a rank range `[rl, rr)`, then
//!    decomposed into O(log n) canonical nodes. Each node's NSW is searched
//!    and the per-node top-k results are merged.
//!
//! Memory: each item appears in O(log(n / leaf_size)) graphs.
//! Search:  ~log(n) graph searches, each tiny.

use crate::nsw::{Nsw, NswParams};
use crate::{sq_l2, Range, RangeAnn};
use std::sync::Arc;

pub struct SegmentGraph {
    pub vectors: Arc<Vec<Vec<f32>>>,
    /// `sorted[r] = (original_id, key)` in ascending key order.
    sorted: Vec<(u32, f32)>,
    /// Segment-tree nodes indexed 1..(2*size). Each node may own an NSW.
    nodes: Vec<Option<Nsw>>,
    size: usize,
    leaf_size: usize,
    params: NswParams,
}

impl SegmentGraph {
    pub fn build(
        vectors: Arc<Vec<Vec<f32>>>,
        keys: Vec<f32>,
        params: NswParams,
        leaf_size: usize,
    ) -> Self {
        assert_eq!(vectors.len(), keys.len());
        let mut sorted: Vec<(u32, f32)> = (0..vectors.len() as u32).map(|i| (i, keys[i as usize])).collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let n = sorted.len();
        let mut size = 1;
        while size < n.max(1) {
            size *= 2;
        }
        let mut nodes: Vec<Option<Nsw>> = (0..2 * size).map(|_| None).collect();

        // Build NSW for every node whose span >= leaf_size and intersects [0, n).
        // Walk recursively; we materialise a node only if it has >= 2 items.
        fn build_node(
            idx: usize,
            l: usize,
            r: usize,
            n: usize,
            leaf_size: usize,
            sorted: &[(u32, f32)],
            vectors: &Arc<Vec<Vec<f32>>>,
            params: NswParams,
            nodes: &mut [Option<Nsw>],
        ) {
            let lo = l.min(n);
            let hi = r.min(n);
            let span = hi.saturating_sub(lo);
            if span == 0 {
                return;
            }
            // Build if this is a leaf-sized subtree OR a small node.
            let is_leaf_node = (r - l) <= leaf_size;
            if is_leaf_node || span <= leaf_size * 2 {
                let ids: Vec<u32> = sorted[lo..hi].iter().map(|(i, _)| *i).collect();
                if ids.len() >= 2 {
                    nodes[idx] = Some(Nsw::build(vectors.clone(), ids, params));
                }
                if is_leaf_node {
                    return;
                }
            } else {
                let ids: Vec<u32> = sorted[lo..hi].iter().map(|(i, _)| *i).collect();
                if ids.len() >= 2 {
                    nodes[idx] = Some(Nsw::build(vectors.clone(), ids, params));
                }
            }
            let mid = (l + r) / 2;
            build_node(2 * idx, l, mid, n, leaf_size, sorted, vectors, params, nodes);
            build_node(2 * idx + 1, mid, r, n, leaf_size, sorted, vectors, params, nodes);
        }

        build_node(1, 0, size, n, leaf_size, &sorted, &vectors, params, &mut nodes);

        Self {
            vectors,
            sorted,
            nodes,
            size,
            leaf_size,
            params,
        }
    }

    /// Rank range `[rl, rr)` containing every item with key in `range`.
    fn key_range_to_rank(&self, range: Range) -> (usize, usize) {
        let rl = self
            .sorted
            .partition_point(|(_, k)| *k < range.lo);
        let rr = self
            .sorted
            .partition_point(|(_, k)| *k <= range.hi);
        (rl, rr)
    }

    /// Iterate canonical segment nodes covering [rl, rr).
    fn canonical_nodes(&self, rl: usize, rr: usize) -> Vec<usize> {
        let mut out = Vec::new();
        self.canon_rec(1, 0, self.size, rl, rr, &mut out);
        out
    }
    fn canon_rec(&self, idx: usize, l: usize, r: usize, rl: usize, rr: usize, out: &mut Vec<usize>) {
        if rr <= l || r <= rl {
            return;
        }
        if rl <= l && r <= rr {
            if self.nodes[idx].is_some() {
                out.push(idx);
                return;
            }
            // No graph at this node: fall through to children.
        }
        if r - l == 1 {
            return;
        }
        let mid = (l + r) / 2;
        self.canon_rec(2 * idx, l, mid, rl, rr, out);
        self.canon_rec(2 * idx + 1, mid, r, rl, rr, out);
    }

    pub fn graph_bytes(&self) -> usize {
        self.nodes
            .iter()
            .filter_map(|o| o.as_ref().map(|g| g.adj_bytes()))
            .sum()
    }

    pub fn num_graphs(&self) -> usize {
        self.nodes.iter().filter(|o| o.is_some()).count()
    }

    /// Brute-force fallback used when the canonical region is below leaf_size
    /// (rare but tidy).
    fn brute(&self, q: &[f32], rl: usize, rr: usize, k: usize) -> Vec<(usize, f32)> {
        let mut hits: Vec<(usize, f32)> = self.sorted[rl..rr]
            .iter()
            .map(|(i, _)| (*i as usize, sq_l2(q, &self.vectors[*i as usize])))
            .collect();
        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        hits.truncate(k);
        hits
    }
}

impl RangeAnn for SegmentGraph {
    fn name(&self) -> &'static str {
        "serf-segment-graph"
    }
    fn search(&self, q: &[f32], range: Range, k: usize) -> Vec<(usize, f32)> {
        let (rl, rr) = self.key_range_to_rank(range);
        if rr <= rl {
            return Vec::new();
        }
        if rr - rl <= self.leaf_size {
            return self.brute(q, rl, rr, k);
        }
        let nodes = self.canonical_nodes(rl, rr);
        let mut merged: Vec<(usize, f32)> = Vec::new();
        let per_node = k.max(self.params.ef_search / 2);
        if nodes.is_empty() {
            return self.brute(q, rl, rr, k);
        }
        for idx in nodes {
            if let Some(g) = &self.nodes[idx] {
                let part = g.search(q, per_node);
                merged.extend(part);
            }
        }
        // Dedup by id (can repeat across nodes if a parent + descendant both used).
        merged.sort_by(|a, b| a.0.cmp(&b.0));
        merged.dedup_by(|a, b| a.0 == b.0);
        merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        merged.truncate(k);
        merged
    }
}
