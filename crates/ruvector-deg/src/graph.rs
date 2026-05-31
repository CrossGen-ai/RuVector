//! Bounded-degree Dynamic Exploration Graph.
//!
//! Layout: vectors are stored row-major in a `Vec<f32>` (`dim * capacity`),
//! and adjacency is a flat `Vec<u32>` with `degree` slots per vertex. A
//! vertex is marked vacant by setting all its neighbour slots to `TOMB`
//! and pushing the index onto a free list. Insertion reuses a free slot
//! before extending the arrays.
//!
//! The core invariant is "exactly `degree` edges per active vertex once
//! the graph has at least `degree + 1` vertices". During the warm-up phase
//! (fewer than `degree + 1` vertices) the graph is fully connected and
//! degrees grow with `len()`.

use crate::distance::Metric;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

pub const TOMB: u32 = u32::MAX;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DegParams {
    /// Out-degree per vertex (D in the paper). Typical: 16–32.
    pub degree: usize,
    /// Search beam width (eps in the paper / `ef` in HNSW). Typical: 30–200.
    pub eps: usize,
    /// Edge-refinement attempts per insert. 0 disables refinement.
    pub refine: usize,
    /// Distance metric.
    pub metric: Metric,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for DegParams {
    fn default() -> Self {
        Self {
            degree: 24,
            eps: 60,
            refine: 4,
            metric: Metric::L2Sq,
            seed: 0xDEC0_DEC0_DEC0_DEC0,
        }
    }
}

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct SearchStats {
    pub distance_calls: u64,
    pub visited: u64,
}

pub struct DegGraph {
    params: DegParams,
    dim: usize,
    /// Row-major vector store: `vectors[id * dim .. (id+1) * dim]`.
    vectors: Vec<f32>,
    /// Flat adjacency: `edges[id * degree .. (id+1) * degree]`. TOMB = unused.
    edges: Vec<u32>,
    /// Edge weight (distance) corresponding to each adjacency slot.
    weights: Vec<f32>,
    /// Active-vertex bitmap (false = vacant slot).
    alive: Vec<bool>,
    /// Free-list of vacant slot ids.
    free: Vec<u32>,
    /// Live vertex count.
    live: usize,
    /// Entry-point id (any live vertex).
    entry: Option<u32>,
    rng: StdRng,
}

impl DegGraph {
    pub fn new(dim: usize, params: DegParams) -> Self {
        let rng = StdRng::seed_from_u64(params.seed);
        Self {
            params,
            dim,
            vectors: Vec::new(),
            edges: Vec::new(),
            weights: Vec::new(),
            alive: Vec::new(),
            free: Vec::new(),
            live: 0,
            entry: None,
            rng,
        }
    }

    pub fn len(&self) -> usize { self.live }
    pub fn is_empty(&self) -> bool { self.live == 0 }
    pub fn capacity(&self) -> usize { self.alive.len() }
    pub fn dim(&self) -> usize { self.dim }
    pub fn params(&self) -> &DegParams { &self.params }

    #[inline]
    fn vec_of(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    #[inline]
    fn edges_of(&self, id: u32) -> &[u32] {
        let start = id as usize * self.params.degree;
        &self.edges[start..start + self.params.degree]
    }

    #[inline]
    fn dist(&self, a: u32, b: &[f32], stats: &mut SearchStats) -> f32 {
        stats.distance_calls += 1;
        self.params.metric.distance(self.vec_of(a), b)
    }

    /// Insert a new vector and return its assigned id.
    pub fn insert(&mut self, vector: &[f32]) -> u32 {
        assert_eq!(vector.len(), self.dim, "vector dim mismatch");

        let new_id = match self.free.pop() {
            Some(id) => {
                let start = id as usize * self.dim;
                self.vectors[start..start + self.dim].copy_from_slice(vector);
                self.alive[id as usize] = true;
                id
            }
            None => {
                let id = self.alive.len() as u32;
                self.vectors.extend_from_slice(vector);
                self.edges.resize(self.edges.len() + self.params.degree, TOMB);
                self.weights.resize(self.weights.len() + self.params.degree, f32::INFINITY);
                self.alive.push(true);
                id
            }
        };
        self.live += 1;

        // Warm-up: fully connect everyone until we have enough vertices.
        if self.live <= self.params.degree + 1 {
            self.rewire_warmup(new_id);
            if self.entry.is_none() { self.entry = Some(new_id); }
            return new_id;
        }

        // Find candidate neighbours via search from the current entry point.
        let mut stats = SearchStats::default();
        let neighbours = self.search_internal(vector, self.params.eps, &mut stats, Some(new_id));

        // Keep top `degree` by ascending distance.
        let mut chosen: Vec<(u32, f32)> = neighbours.into_iter().take(self.params.degree).collect();
        while chosen.len() < self.params.degree {
            // Pad with random live vertices if we somehow got too few.
            if let Some(rid) = self.random_live(new_id) {
                let d = self.params.metric.distance(self.vec_of(rid), vector);
                chosen.push((rid, d));
            } else {
                break;
            }
        }

        // Install forward edges.
        {
            let deg = self.params.degree;
            let estart = new_id as usize * deg;
            for (slot, (nbr, w)) in chosen.iter().enumerate().take(deg) {
                self.edges[estart + slot] = *nbr;
                self.weights[estart + slot] = *w;
            }
            for slot in chosen.len()..deg {
                self.edges[estart + slot] = TOMB;
                self.weights[estart + slot] = f32::INFINITY;
            }
        }

        // Reverse-link: ask each neighbour to consider adopting new_id, evicting
        // its current heaviest edge if new_id is closer. This preserves the
        // bounded-degree invariant.
        for (nbr, w) in &chosen {
            self.try_adopt(*nbr, new_id, *w);
        }

        // Edge refinement: try `refine` random triangle improvements.
        for _ in 0..self.params.refine {
            self.refine_once(new_id);
        }

        if self.entry.is_none() { self.entry = Some(new_id); }
        new_id
    }

    /// Delete a previously-inserted id.
    pub fn delete(&mut self, id: u32) {
        if (id as usize) >= self.alive.len() || !self.alive[id as usize] {
            return;
        }
        // Patch every vertex that pointed at `id` by re-searching with that
        // vertex's own vector and adopting its closest non-dead non-self
        // candidate that isn't already a neighbour.
        let referrers: Vec<u32> = (0..self.alive.len() as u32)
            .filter(|&v| self.alive[v as usize] && v != id)
            .filter(|&v| self.edges_of(v).contains(&id))
            .collect();

        for v in referrers {
            self.patch_edge(v, id);
        }

        // Mark vacant.
        let estart = id as usize * self.params.degree;
        for slot in 0..self.params.degree {
            self.edges[estart + slot] = TOMB;
            self.weights[estart + slot] = f32::INFINITY;
        }
        self.alive[id as usize] = false;
        self.free.push(id);
        self.live -= 1;

        if self.entry == Some(id) {
            self.entry = (0..self.alive.len() as u32).find(|&v| self.alive[v as usize]);
        }
    }

    /// k-NN query. Returns `(id, distance)` pairs sorted ascending.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let mut stats = SearchStats::default();
        let mut res = self.search_internal(query, k.max(self.params.eps), &mut stats, None);
        res.truncate(k);
        res
    }

    pub fn search_with_stats(&self, query: &[f32], k: usize) -> (Vec<(u32, f32)>, SearchStats) {
        let mut stats = SearchStats::default();
        let mut res = self.search_internal(query, k.max(self.params.eps), &mut stats, None);
        res.truncate(k);
        (res, stats)
    }

    /// Best-first search returning the top `beam` closest vertices.
    /// `exclude` is set when the query vector is itself being inserted, so
    /// we don't return the not-yet-finalised vertex as a neighbour of itself.
    fn search_internal(
        &self,
        query: &[f32],
        beam: usize,
        stats: &mut SearchStats,
        exclude: Option<u32>,
    ) -> Vec<(u32, f32)> {
        let Some(entry) = self.entry else { return Vec::new(); };

        let mut visited: HashSet<u32> = HashSet::with_capacity(beam * 4);
        // Min-heap of frontier (closest first → wrap with `Reverse` via sign flip).
        let mut frontier: BinaryHeap<Candidate> = BinaryHeap::new();
        // Max-heap of best (so we can pop the worst when over capacity).
        let mut best: BinaryHeap<WorstFirst> = BinaryHeap::new();

        let d0 = self.params.metric.distance(self.vec_of(entry), query);
        stats.distance_calls += 1;
        frontier.push(Candidate { id: entry, dist: d0 });
        best.push(WorstFirst { id: entry, dist: d0 });
        visited.insert(entry);

        while let Some(Candidate { id: cur, dist: d_cur }) = frontier.pop() {
            stats.visited += 1;
            // Termination: if the closest unexplored is farther than the worst
            // in `best`, we can stop.
            if let Some(worst) = best.peek() {
                if best.len() >= beam && d_cur > worst.dist {
                    break;
                }
            }
            for &nbr in self.edges_of(cur) {
                if nbr == TOMB { continue; }
                if !self.alive[nbr as usize] { continue; }
                if Some(nbr) == exclude { continue; }
                if !visited.insert(nbr) { continue; }
                let d = self.dist(nbr, query, stats);
                if best.len() < beam {
                    best.push(WorstFirst { id: nbr, dist: d });
                    frontier.push(Candidate { id: nbr, dist: d });
                } else if d < best.peek().unwrap().dist {
                    best.pop();
                    best.push(WorstFirst { id: nbr, dist: d });
                    frontier.push(Candidate { id: nbr, dist: d });
                }
            }
        }

        let mut out: Vec<(u32, f32)> =
            best.into_iter().map(|w| (w.id, w.dist)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        out
    }

    /// Connect `new_id` to every other live vertex (warm-up phase).
    fn rewire_warmup(&mut self, new_id: u32) {
        let mut pairs: Vec<(u32, f32)> = (0..self.alive.len() as u32)
            .filter(|&v| v != new_id && self.alive[v as usize])
            .map(|v| {
                let d = self.params.metric.distance(self.vec_of(v), self.vec_of(new_id));
                (v, d)
            })
            .collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        let deg = self.params.degree;
        let estart = new_id as usize * deg;
        for slot in 0..deg {
            if slot < pairs.len() {
                self.edges[estart + slot] = pairs[slot].0;
                self.weights[estart + slot] = pairs[slot].1;
            } else {
                self.edges[estart + slot] = TOMB;
                self.weights[estart + slot] = f32::INFINITY;
            }
        }
        for (nbr, w) in pairs {
            self.try_adopt(nbr, new_id, w);
        }
    }

    /// Offer `candidate` (with known distance `w`) to vertex `host`. If `host`
    /// has a vacant slot or a heavier edge, replace.
    fn try_adopt(&mut self, host: u32, candidate: u32, w: f32) {
        if host == candidate || !self.alive[host as usize] {
            return;
        }
        let deg = self.params.degree;
        let estart = host as usize * deg;
        // Already a neighbour?
        for slot in 0..deg {
            if self.edges[estart + slot] == candidate { return; }
        }
        // Find vacant first.
        for slot in 0..deg {
            if self.edges[estart + slot] == TOMB {
                self.edges[estart + slot] = candidate;
                self.weights[estart + slot] = w;
                return;
            }
        }
        // Else evict heaviest if w is smaller.
        let mut worst_slot = 0usize;
        let mut worst_w = self.weights[estart];
        for slot in 1..deg {
            if self.weights[estart + slot] > worst_w {
                worst_w = self.weights[estart + slot];
                worst_slot = slot;
            }
        }
        if w < worst_w {
            self.edges[estart + worst_slot] = candidate;
            self.weights[estart + worst_slot] = w;
        }
    }

    /// Re-search from `host`'s perspective to replace its edge that pointed
    /// to `dead`. The patch tries to find a vertex that is *closer* than the
    /// host's current heaviest remaining edge, otherwise picks the next best
    /// available.
    fn patch_edge(&mut self, host: u32, dead: u32) {
        let host_vec = self.vec_of(host).to_vec();
        let mut stats = SearchStats::default();
        // Search excluding `dead` (and `host` itself).
        let cands = self.search_internal(&host_vec, self.params.eps, &mut stats, Some(host));
        let neighbours_now: HashSet<u32> = self.edges_of(host).iter().copied().collect();

        let mut replacement: Option<(u32, f32)> = None;
        for (cand, d) in cands {
            if cand == host || cand == dead { continue; }
            if neighbours_now.contains(&cand) { continue; }
            replacement = Some((cand, d));
            break;
        }

        let deg = self.params.degree;
        let estart = host as usize * deg;
        for slot in 0..deg {
            if self.edges[estart + slot] == dead {
                if let Some((nbr, w)) = replacement {
                    self.edges[estart + slot] = nbr;
                    self.weights[estart + slot] = w;
                } else {
                    self.edges[estart + slot] = TOMB;
                    self.weights[estart + slot] = f32::INFINITY;
                }
                return;
            }
        }
    }

    /// One refinement attempt: pick a random edge of `seed`, look at the
    /// neighbour's neighbours, and see if any of them gives a strictly
    /// shorter outgoing edge for `seed`.
    fn refine_once(&mut self, seed: u32) {
        let deg = self.params.degree;
        let estart = seed as usize * deg;
        let pick_slot = self.rng.gen_range(0..deg);
        let nbr = self.edges[estart + pick_slot];
        if nbr == TOMB { return; }
        // Snapshot neighbour's neighbours to avoid borrow conflicts.
        let candidates: Vec<u32> = self.edges_of(nbr).to_vec();
        let seed_vec = self.vec_of(seed).to_vec();
        let neighbours_now: HashSet<u32> = self.edges_of(seed).iter().copied().collect();

        // Find heaviest current outgoing edge of seed.
        let mut worst_slot = 0usize;
        let mut worst_w = self.weights[estart];
        for slot in 1..deg {
            if self.weights[estart + slot] > worst_w {
                worst_w = self.weights[estart + slot];
                worst_slot = slot;
            }
        }

        for c in candidates {
            if c == TOMB || c == seed { continue; }
            if !self.alive[c as usize] { continue; }
            if neighbours_now.contains(&c) { continue; }
            let d = self.params.metric.distance(self.vec_of(c), &seed_vec);
            if d < worst_w {
                self.edges[estart + worst_slot] = c;
                self.weights[estart + worst_slot] = d;
                // And reverse-link.
                self.try_adopt(c, seed, d);
                return;
            }
        }
    }

    fn random_live(&mut self, exclude: u32) -> Option<u32> {
        let live: Vec<u32> = (0..self.alive.len() as u32)
            .filter(|&v| v != exclude && self.alive[v as usize])
            .collect();
        live.choose(&mut self.rng).copied()
    }

    /// Total live edges (for sanity diagnostics).
    pub fn edge_count(&self) -> usize {
        let mut n = 0usize;
        for v in 0..self.alive.len() as u32 {
            if !self.alive[v as usize] { continue; }
            for &e in self.edges_of(v) {
                if e != TOMB { n += 1; }
            }
        }
        n
    }

    /// Mean outgoing edge weight across live vertices (proxy for graph
    /// quality — lower is better).
    pub fn mean_edge_weight(&self) -> f64 {
        let mut s = 0.0f64;
        let mut n = 0usize;
        for v in 0..self.alive.len() as u32 {
            if !self.alive[v as usize] { continue; }
            let estart = v as usize * self.params.degree;
            for slot in 0..self.params.degree {
                let w = self.weights[estart + slot];
                if w.is_finite() {
                    s += w as f64;
                    n += 1;
                }
            }
        }
        if n == 0 { 0.0 } else { s / n as f64 }
    }
}

// --- heap helpers --- //

#[derive(Copy, Clone, Debug)]
struct Candidate { id: u32, dist: f32 }
impl Eq for Candidate {}
impl PartialEq for Candidate { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Ord for Candidate {
    // Smaller dist => greater priority (min-heap via reverse).
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Candidate { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }

#[derive(Copy, Clone, Debug)]
struct WorstFirst { id: u32, dist: f32 }
impl Eq for WorstFirst {}
impl PartialEq for WorstFirst { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Ord for WorstFirst {
    // Larger dist => greater priority (max-heap directly).
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.partial_cmp(&o.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for WorstFirst { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
