//! Minimal single-layer NSW (Navigable Small World) index.
//!
//! This is intentionally compact: the goal of the crate is to study
//! `ef_search` adaptation, not to outperform `hnswlib`. The graph is built by
//! incremental insertion with the same `ef_construction` beam search HNSW
//! uses on its bottom layer.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Cosine-prepared dense f32 vector (rows L2-normalised so we can use
/// squared-Euclidean as a monotone proxy for cosine distance — the choice
/// between the two does not affect `ef_search` adaptation behaviour).
pub type Vector = Vec<f32>;

/// Distance between two equal-length vectors. Squared L2.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Per-query bookkeeping returned alongside top-k. Used to measure work.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchStats {
    pub distance_computations: u64,
    pub visited_nodes: u64,
    pub ef_used: u32,
}

#[derive(Clone, Copy)]
struct Candidate {
    id: u32,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we want closest = smallest distance to
        // pop first when used as a min-heap (so flip), or largest when used
        // as the top-k max-heap.
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Wrappers to make min/max heap semantics explicit and avoid bugs.
#[derive(Clone, Copy)]
struct Closer(Candidate); // min-heap on dist (Reverse)
impl PartialEq for Closer {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for Closer {}
impl Ord for Closer {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.dist.partial_cmp(&self.0.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Closer {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[derive(Clone, Copy)]
struct Farther(Candidate); // max-heap on dist
impl PartialEq for Farther {
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl Eq for Farther {}
impl Ord for Farther {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.dist.partial_cmp(&o.0.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Farther {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Build/search parameters for [`NswIndex`].
#[derive(Debug, Clone, Copy)]
pub struct NswParams {
    pub m: usize,               // edges per node
    pub ef_construction: usize, // beam width during build
    pub dim: usize,
}

impl NswParams {
    pub fn new(dim: usize) -> Self {
        Self { m: 32, ef_construction: 200, dim }
    }
}

/// Single-layer NSW graph.
pub struct NswIndex {
    pub params: NswParams,
    vectors: Vec<Vector>,
    edges: Vec<Vec<u32>>,
    entry: Option<u32>,
}

impl NswIndex {
    pub fn new(params: NswParams) -> Self {
        Self { params, vectors: Vec::new(), edges: Vec::new(), entry: None }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    pub fn vector(&self, id: u32) -> &[f32] {
        &self.vectors[id as usize]
    }

    /// Insert `v` and connect it to up to `m` neighbours found via beam search.
    pub fn insert(&mut self, v: Vector) -> u32 {
        assert_eq!(v.len(), self.params.dim);
        let id = self.vectors.len() as u32;
        self.vectors.push(v);
        self.edges.push(Vec::with_capacity(self.params.m));

        if self.entry.is_none() {
            self.entry = Some(id);
            return id;
        }

        let entry = self.entry.unwrap();
        let v_ref = &self.vectors[id as usize];
        let mut stats = SearchStats::default();
        let cands = self.search_layer(
            v_ref,
            entry,
            self.params.ef_construction,
            &mut stats,
        );

        // pick up to M closest as neighbours, add reciprocal edges
        let mut chosen: Vec<u32> = cands.into_iter().take(self.params.m).map(|c| c.id).collect();
        for &nbr in &chosen {
            // bidirectional, prune over-full neighbours by farthest distance
            let nbr_idx = nbr as usize;
            self.edges[nbr_idx].push(id);
            if self.edges[nbr_idx].len() > self.params.m {
                self.shrink_neighbours(nbr);
            }
        }
        // Pull our own list
        self.edges[id as usize].append(&mut chosen);

        id
    }

    fn shrink_neighbours(&mut self, node: u32) {
        let v = self.vectors[node as usize].clone();
        let mut scored: Vec<Candidate> = self.edges[node as usize]
            .iter()
            .map(|&n| Candidate { id: n, dist: sq_l2(&v, &self.vectors[n as usize]) })
            .collect();
        scored.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        scored.truncate(self.params.m);
        self.edges[node as usize] = scored.into_iter().map(|c| c.id).collect();
    }

    /// Greedy + beam search at a single layer. Returns candidates sorted by
    /// ascending distance. Updates `stats` in place.
    fn search_layer(
        &self,
        query: &[f32],
        entry: u32,
        ef: usize,
        stats: &mut SearchStats,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        let mut frontier: BinaryHeap<Closer> = BinaryHeap::new();
        let mut top: BinaryHeap<Farther> = BinaryHeap::new();

        let entry_d = sq_l2(query, &self.vectors[entry as usize]);
        stats.distance_computations += 1;
        let entry_c = Candidate { id: entry, dist: entry_d };
        frontier.push(Closer(entry_c));
        top.push(Farther(entry_c));
        visited.insert(entry);

        while let Some(Closer(cur)) = frontier.pop() {
            // worst element in top determines our pruning bound
            let bound = top.peek().map(|f| f.0.dist).unwrap_or(f32::INFINITY);
            if cur.dist > bound && top.len() >= ef {
                break;
            }
            stats.visited_nodes += 1;

            for &nbr in &self.edges[cur.id as usize] {
                if !visited.insert(nbr) {
                    continue;
                }
                let d = sq_l2(query, &self.vectors[nbr as usize]);
                stats.distance_computations += 1;
                let nc = Candidate { id: nbr, dist: d };
                if top.len() < ef {
                    frontier.push(Closer(nc));
                    top.push(Farther(nc));
                } else if d < top.peek().unwrap().0.dist {
                    frontier.push(Closer(nc));
                    top.push(Farther(nc));
                    while top.len() > ef {
                        top.pop();
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = top.into_iter().map(|f| f.0).collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out
    }

    /// Public search: returns top-k ids and per-query stats.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> (Vec<u32>, SearchStats) {
        let mut stats = SearchStats { ef_used: ef as u32, ..Default::default() };
        let Some(entry) = self.entry else {
            return (Vec::new(), stats);
        };
        let ef_use = ef.max(k);
        let cands = self.search_layer(query, entry, ef_use, &mut stats);
        let ids: Vec<u32> = cands.into_iter().take(k).map(|c| c.id).collect();
        (ids, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn rand_unit(rng: &mut StdRng, d: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        for x in &mut v { *x /= n; }
        v
    }

    #[test]
    fn recall_grows_with_ef() {
        let mut rng = StdRng::seed_from_u64(7);
        let d = 32;
        let mut idx = NswIndex::new(NswParams::new(d));
        let vecs: Vec<Vec<f32>> = (0..1500).map(|_| rand_unit(&mut rng, d)).collect();
        for v in &vecs { idx.insert(v.clone()); }

        let queries: Vec<Vec<f32>> = (0..50).map(|_| rand_unit(&mut rng, d)).collect();
        let k = 10;

        let mut last = 0.0;
        for ef in [10, 32, 128] {
            let mut hits = 0;
            let mut total = 0;
            for q in &queries {
                let mut gt: Vec<(u32, f32)> = vecs.iter().enumerate()
                    .map(|(i, v)| (i as u32, sq_l2(q, v))).collect();
                gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                let gt_ids: std::collections::HashSet<u32> =
                    gt.into_iter().take(k).map(|x| x.0).collect();
                let (got, _) = idx.search(q, k, ef);
                hits += got.iter().filter(|i| gt_ids.contains(*i)).count();
                total += k;
            }
            let recall = hits as f64 / total as f64;
            assert!(recall + 1e-9 >= last, "recall should be monotone in ef");
            last = recall;
        }
        assert!(last > 0.7, "ef=128 should reach reasonable recall, got {last}");
    }
}
