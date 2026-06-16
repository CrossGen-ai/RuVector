//! Segment-tree of proximity graphs for range-filtered ANN.
//!
//! Inspired by iRangeGraph (SIGMOD 2024). Points are sorted by the range key,
//! then a segment tree is built where each node owns a Graph over its slice of
//! contiguous points. A query [lo, hi] is covered by O(log n) tree nodes whose
//! key intervals are fully contained in [lo, hi]; the boundary nodes (partial
//! overlap) fall back to a leaf-level linear scan over the relevant points.

use crate::graph::Graph;

pub struct SegmentRangeIndex {
    points_sorted: Vec<(u32, f32, Vec<f32>)>, // (global_id, key, vector)
    nodes: Vec<Option<Graph>>,                 // 1-indexed segment tree; root at 1
    leaf_size: usize,
    n: usize,
    m: usize,
    ef_construction: usize,
    range_min: f32,
    range_max: f32,
}

impl SegmentRangeIndex {
    pub fn build(
        mut data: Vec<(u32, f32, Vec<f32>)>,
        leaf_size: usize,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let n = data.len();
        let range_min = data.first().map(|x| x.1).unwrap_or(0.0);
        let range_max = data.last().map(|x| x.1).unwrap_or(0.0);

        // Segment-tree size: next power of two of ceil(n/leaf_size).
        let leaves = (n + leaf_size - 1) / leaf_size.max(1);
        let mut tree_size = 1usize;
        while tree_size < leaves.max(1) { tree_size *= 2; }
        let total_nodes = 2 * tree_size;
        let nodes: Vec<Option<Graph>> = (0..total_nodes).map(|_| None).collect();

        let mut s = SegmentRangeIndex {
            points_sorted: data,
            nodes,
            leaf_size,
            n,
            m,
            ef_construction,
            range_min,
            range_max,
        };
        s.build_node(1, 0, n);
        s
    }

    /// Recursive build: node `idx` owns the slice `[lo, hi)` of `points_sorted`.
    fn build_node(&mut self, idx: usize, lo: usize, hi: usize) {
        if lo >= hi { return; }
        let len = hi - lo;
        let slice = &self.points_sorted[lo..hi];
        let vectors: Vec<Vec<f32>> = slice.iter().map(|(_, _, v)| v.clone()).collect();
        let gids: Vec<u32> = slice.iter().map(|(g, _, _)| *g).collect();
        let keys: Vec<f32> = slice.iter().map(|(_, k, _)| *k).collect();
        let g = Graph::build(vectors, gids, keys, self.m, self.ef_construction);
        if idx < self.nodes.len() {
            self.nodes[idx] = Some(g);
        }
        if len <= self.leaf_size { return; }
        let mid = lo + len / 2;
        self.build_node(idx * 2, lo, mid);
        self.build_node(idx * 2 + 1, mid, hi);
    }

    /// Range-filtered top-k search: returns `(distance², global_id, key)`,
    /// sorted by distance ascending.
    pub fn search(&self, q: &[f32], k: usize, lo: f32, hi: f32, ef: usize) -> Vec<(f32, u32, f32)> {
        let mut hits: Vec<(f32, u32, f32)> = Vec::new();
        self.query(1, 0, self.n, lo, hi, q, ef, &mut hits);
        hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        hits.dedup_by_key(|x| x.1);
        hits.truncate(k);
        hits
    }

    fn query(
        &self,
        idx: usize,
        node_lo: usize,
        node_hi: usize,
        lo: f32,
        hi: f32,
        q: &[f32],
        ef: usize,
        out: &mut Vec<(f32, u32, f32)>,
    ) {
        if node_lo >= node_hi { return; }
        let key_lo = self.points_sorted[node_lo].1;
        let key_hi = self.points_sorted[node_hi - 1].1;
        if key_hi < lo || key_lo > hi { return; }
        // Fully covered: run mini-graph search on this node.
        if key_lo >= lo && key_hi <= hi {
            if let Some(g) = self.nodes.get(idx).and_then(|n| n.as_ref()) {
                let res = g.search(q, ef);
                out.extend(res);
            }
            return;
        }
        // Leaf with partial overlap: brute force the slice.
        let len = node_hi - node_lo;
        if len <= self.leaf_size {
            for (gid, k, v) in &self.points_sorted[node_lo..node_hi] {
                if (lo..=hi).contains(k) {
                    let d = crate::dist::l2_sq(q, v);
                    out.push((d, *gid, *k));
                }
            }
            return;
        }
        let mid = node_lo + len / 2;
        self.query(idx * 2, node_lo, mid, lo, hi, q, ef, out);
        self.query(idx * 2 + 1, mid, node_hi, lo, hi, q, ef, out);
    }

    pub fn memory_bytes(&self) -> usize {
        let trees: usize = self.nodes.iter().filter_map(|n| n.as_ref()).map(|g| g.memory_bytes()).sum();
        let raw = self.points_sorted.iter().map(|(_, _, v)| v.capacity() * 4 + 8).sum::<usize>();
        trees + raw
    }

    pub fn len(&self) -> usize { self.n }
    pub fn is_empty(&self) -> bool { self.n == 0 }
    pub fn range(&self) -> (f32, f32) { (self.range_min, self.range_max) }
}
