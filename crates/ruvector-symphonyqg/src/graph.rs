//! Small NSW-style proximity graph used as the substrate for SymphonyQG.
//!
//! Build:  for each inserted point, greedy-search the current graph for the
//! `ef_construction` closest points, then keep the `m` nearest as edges and
//! back-link from each chosen neighbour (capped at `m`).
//!
//! This is deliberately a flat (1-layer) NSW, not full HNSW: the point of the
//! paper is the graph/quantization interplay, not multi-layer routing. It
//! suffices for the recall/QPS comparison and keeps the file readable.

use crate::l2_sq;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A single edge target.
type NodeId = u32;

/// Priority-queue entry, ordered by ascending distance (min-heap via `Reverse`).
#[derive(Copy, Clone, Debug)]
struct Cand {
    dist: f32,
    id: NodeId,
}
impl Eq for Cand {}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool { self.dist == o.dist && self.id == o.id }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        // NaN-safe: treat unequal floats via total_cmp.
        self.dist.total_cmp(&o.dist).then(self.id.cmp(&o.id))
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}

/// A simple flat proximity graph. Owns the vectors and the adjacency lists.
pub struct Graph {
    pub dim: usize,
    pub m: usize,
    pub vectors: Vec<f32>,        // row-major, len = n * dim
    pub adj: Vec<Vec<NodeId>>,    // adj[i] = neighbours of node i
    pub entry: NodeId,
}

impl Graph {
    pub fn len(&self) -> usize { self.adj.len() }
    pub fn is_empty(&self) -> bool { self.adj.is_empty() }

    #[inline]
    pub fn vec_of(&self, id: NodeId) -> &[f32] {
        let s = (id as usize) * self.dim;
        &self.vectors[s..s + self.dim]
    }

    /// Greedy beam search returning the closest `k` nodes (by float L2).
    /// `ef` is the candidate-set width.
    pub fn search_float(&self, query: &[f32], k: usize, ef: usize) -> Vec<(f32, NodeId)> {
        self.search_with(query, k, ef, |id| l2_sq(query, self.vec_of(id)))
    }

    /// Generic graph traversal with a caller-supplied scoring function.
    /// Returns top-`k` (asc dist) of the visited frontier.
    pub fn search_with<F>(&self, _q: &[f32], k: usize, ef: usize, mut score: F)
        -> Vec<(f32, NodeId)>
    where
        F: FnMut(NodeId) -> f32,
    {
        let n = self.len();
        if n == 0 { return Vec::new(); }
        let mut visited = vec![false; n];

        let mut frontier: BinaryHeap<std::cmp::Reverse<Cand>> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new(); // max-heap of size <= ef

        // Multi-entry seeding: the static entry plus a deterministic spread of
        // points across the dataset. Cures the "single-layer NSW stuck in one
        // cluster" failure mode without paying for an HNSW upper layer.
        let n_seeds = 8.min(n);
        let mut seeds: Vec<NodeId> = Vec::with_capacity(n_seeds);
        seeds.push(self.entry);
        if n > 1 {
            let step = (n as u64).max(1);
            for s in 1..n_seeds {
                let id = ((s as u64) * step / n_seeds as u64) as NodeId % (n as NodeId);
                if id != self.entry { seeds.push(id); }
            }
        }
        for id in seeds {
            let i = id as usize;
            if visited[i] { continue; }
            visited[i] = true;
            let d = score(id);
            frontier.push(std::cmp::Reverse(Cand { dist: d, id }));
            results.push(Cand { dist: d, id });
            if results.len() > ef { results.pop(); }
        }

        while let Some(std::cmp::Reverse(cur)) = frontier.pop() {
            // Early-stop when the closest frontier candidate is worse than the
            // worst kept result and we already have `ef` results — classic NSW.
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if cur.dist > worst.dist { break; }
                }
            }
            for &nb in &self.adj[cur.id as usize] {
                let i = nb as usize;
                if visited[i] { continue; }
                visited[i] = true;
                let d = score(nb);
                let push = if results.len() < ef {
                    true
                } else {
                    results.peek().map(|w| d < w.dist).unwrap_or(false)
                };
                if push {
                    frontier.push(std::cmp::Reverse(Cand { dist: d, id: nb }));
                    results.push(Cand { dist: d, id: nb });
                    if results.len() > ef { results.pop(); }
                }
            }
        }

        let mut out: Vec<Cand> = results.into_sorted_vec();
        out.truncate(k);
        out.into_iter().map(|c| (c.dist, c.id)).collect()
    }
}

/// Streaming builder: insert one vector at a time.
pub struct GraphBuilder {
    dim: usize,
    m: usize,
    ef_construction: usize,
    vectors: Vec<f32>,
    adj: Vec<Vec<NodeId>>,
    entry: Option<NodeId>,
}

impl GraphBuilder {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        assert!(dim > 0 && m > 0 && ef_construction >= m);
        Self { dim, m, ef_construction, vectors: Vec::new(), adj: Vec::new(), entry: None }
    }

    pub fn insert(&mut self, v: &[f32]) -> NodeId {
        assert_eq!(v.len(), self.dim);
        let id = self.adj.len() as NodeId;
        self.vectors.extend_from_slice(v);
        self.adj.push(Vec::with_capacity(self.m));

        if self.entry.is_none() {
            self.entry = Some(id);
            return id;
        }

        // Reuse Graph's search by constructing a temporary view of what we have.
        // Cheap: shares the same backing buffers; we just need an immutable handle.
        let entry = self.entry.unwrap();
        let view = GraphView {
            dim: self.dim,
            vectors: &self.vectors,
            adj: &self.adj,
            entry,
        };
        let cands = view.search_float(v, self.m, self.ef_construction);

        // Wire forward edges.
        for (_, nb) in &cands {
            self.adj[id as usize].push(*nb);
        }
        // Wire reverse edges with cap-`m` pruning (keep closest).
        for &(_, nb) in &cands {
            let i = nb as usize;
            if self.adj[i].len() < self.m {
                self.adj[i].push(id);
            } else {
                // Replace farthest neighbour if `id` is closer than at least one.
                let mut worst_j = 0;
                let mut worst_d = -1.0f32;
                let nb_vec_start = i * self.dim;
                let nb_vec = &self.vectors[nb_vec_start..nb_vec_start + self.dim];
                for (j, &other) in self.adj[i].iter().enumerate() {
                    let o = other as usize;
                    let d = l2_sq(nb_vec, &self.vectors[o * self.dim..(o + 1) * self.dim]);
                    if d > worst_d { worst_d = d; worst_j = j; }
                }
                let new_d = l2_sq(nb_vec, v);
                if new_d < worst_d {
                    self.adj[i][worst_j] = id;
                }
            }
        }
        id
    }

    pub fn build(self) -> Graph {
        Graph {
            dim: self.dim,
            m: self.m,
            vectors: self.vectors,
            adj: self.adj,
            entry: self.entry.unwrap_or(0),
        }
    }
}

/// Read-only view used during insertion (avoids cloning vectors).
struct GraphView<'a> {
    dim: usize,
    vectors: &'a [f32],
    adj: &'a [Vec<NodeId>],
    entry: NodeId,
}

impl<'a> GraphView<'a> {
    fn vec_of(&self, id: NodeId) -> &[f32] {
        let s = (id as usize) * self.dim;
        &self.vectors[s..s + self.dim]
    }

    fn search_float(&self, query: &[f32], k: usize, ef: usize) -> Vec<(f32, NodeId)> {
        let n = self.adj.len();
        if n == 0 { return Vec::new(); }
        let mut visited = vec![false; n];

        let mut frontier: BinaryHeap<std::cmp::Reverse<Cand>> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();
        // Multi-entry seeding (mirrors Graph::search_with).
        let n_seeds = 8.min(n);
        let mut seeds: Vec<NodeId> = Vec::with_capacity(n_seeds);
        seeds.push(self.entry);
        if n > 1 {
            for s in 1..n_seeds {
                let id = ((s as u64) * n as u64 / n_seeds as u64) as NodeId;
                if id != self.entry { seeds.push(id); }
            }
        }
        for id in seeds {
            let i = id as usize;
            if visited[i] { continue; }
            visited[i] = true;
            let d = l2_sq(query, self.vec_of(id));
            frontier.push(std::cmp::Reverse(Cand { dist: d, id }));
            results.push(Cand { dist: d, id });
            if results.len() > ef { results.pop(); }
        }

        while let Some(std::cmp::Reverse(cur)) = frontier.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if cur.dist > worst.dist { break; }
                }
            }
            for &nb in &self.adj[cur.id as usize] {
                let i = nb as usize;
                if visited[i] { continue; }
                visited[i] = true;
                let d = l2_sq(query, self.vec_of(nb));
                let push = if results.len() < ef {
                    true
                } else {
                    results.peek().map(|w| d < w.dist).unwrap_or(false)
                };
                if push {
                    frontier.push(std::cmp::Reverse(Cand { dist: d, id: nb }));
                    results.push(Cand { dist: d, id: nb });
                    if results.len() > ef { results.pop(); }
                }
            }
        }

        let mut out: Vec<Cand> = results.into_sorted_vec();
        out.truncate(k);
        out.into_iter().map(|c| (c.dist, c.id)).collect()
    }
}
