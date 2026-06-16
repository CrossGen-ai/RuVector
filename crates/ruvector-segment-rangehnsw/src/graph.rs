//! Single-layer proximity graph with greedy beam search.
//!
//! This is intentionally minimal so the PoC stays under the 500-LOC ceiling and
//! all variants share the same primitive. Construction is NN-descent-lite: each
//! node is connected to the `m` approximate nearest neighbors found by greedy
//! search from a random entry point. Search is a standard best-first beam.

use crate::dist::l2_sq;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

/// `local_id` is the index inside the underlying slice that the graph indexes.
pub struct Graph {
    pub m: usize,
    pub ef_construction: usize,
    pub neighbors: Vec<Vec<u32>>, // adjacency lists, indexed by local id
    pub vectors: Vec<Vec<f32>>,   // owned copies of vectors (PoC: keeps API simple)
    pub global_ids: Vec<u32>,     // local id -> caller-supplied global id
    pub keys: Vec<f32>,           // local id -> range key (e.g. timestamp)
    entry: u32,
}

#[derive(Copy, Clone)]
struct OrdF32(f32, u32);
impl Eq for OrdF32 {}
impl PartialEq for OrdF32 { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // primary by distance, then by id (deterministic tiebreak)
        self.0.partial_cmp(&o.0).unwrap_or(std::cmp::Ordering::Equal)
            .then(self.1.cmp(&o.1))
    }
}
impl PartialOrd for OrdF32 { fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(o)) } }

impl Graph {
    pub fn build(
        vectors: Vec<Vec<f32>>,
        global_ids: Vec<u32>,
        keys: Vec<f32>,
        m: usize,
        ef_construction: usize,
    ) -> Self {
        let n = vectors.len();
        let mut g = Graph {
            m,
            ef_construction,
            neighbors: vec![Vec::new(); n],
            vectors,
            global_ids,
            keys,
            entry: 0,
        };
        if n == 0 { return g; }
        // Insert one by one; entry point starts at 0.
        for i in 1..n {
            let i_u = i as u32;
            // Find approximate top-ef neighbors from entry.
            let cands = g.search_internal(&g.vectors[i].clone(), g.ef_construction, &[i_u]);
            // Take top-m as neighbors and add reverse edges.
            let chosen: Vec<u32> = cands.iter().take(m).map(|x| x.1).collect();
            for &nb in &chosen {
                if !g.neighbors[i].contains(&nb) {
                    g.neighbors[i].push(nb);
                }
                if !g.neighbors[nb as usize].contains(&i_u) {
                    g.neighbors[nb as usize].push(i_u);
                    if g.neighbors[nb as usize].len() > 2 * m {
                        // Trim back to the closest m+1 to keep degree bounded.
                        let v_nb = g.vectors[nb as usize].clone();
                        g.neighbors[nb as usize].sort_by(|&a, &b| {
                            let da = l2_sq(&v_nb, &g.vectors[a as usize]);
                            let db = l2_sq(&v_nb, &g.vectors[b as usize]);
                            da.partial_cmp(&db).unwrap()
                        });
                        g.neighbors[nb as usize].truncate(m + 1);
                    }
                }
            }
        }
        g
    }

    /// Beam search. Returns (dist, local_id) pairs sorted by dist asc.
    /// `excluded` is small (used to skip self during build).
    pub fn search_internal(&self, q: &[f32], ef: usize, excluded: &[u32]) -> Vec<(f32, u32)> {
        if self.vectors.is_empty() { return Vec::new(); }
        let mut visited: Vec<bool> = vec![false; self.vectors.len()];
        let start = self.entry as usize;
        visited[start] = true;
        let d0 = l2_sq(q, &self.vectors[start]);
        // candidates: min-heap by distance (closer first); results: max-heap (we cap at ef).
        let mut candidates: BinaryHeap<Reverse<OrdF32>> = BinaryHeap::new();
        let mut results: BinaryHeap<OrdF32> = BinaryHeap::new();
        candidates.push(Reverse(OrdF32(d0, start as u32)));
        if !excluded.contains(&(start as u32)) {
            results.push(OrdF32(d0, start as u32));
        }
        while let Some(Reverse(c)) = candidates.pop() {
            let worst_in_results = results.peek().map(|x| x.0).unwrap_or(f32::INFINITY);
            if results.len() >= ef && c.0 > worst_in_results { break; }
            for &nb in &self.neighbors[c.1 as usize] {
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let d = l2_sq(q, &self.vectors[nb as usize]);
                let worst = results.peek().map(|x| x.0).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(Reverse(OrdF32(d, nb)));
                    if !excluded.contains(&nb) {
                        results.push(OrdF32(d, nb));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }
        let mut out: Vec<(f32, u32)> = results.into_iter().map(|x| (x.0, x.1)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    /// Search returning (distance, global_id, key) so callers can filter by
    /// range without an extra lookup.
    pub fn search(&self, q: &[f32], ef: usize) -> Vec<(f32, u32, f32)> {
        self.search_internal(q, ef, &[])
            .into_iter()
            .map(|(d, lid)| (d, self.global_ids[lid as usize], self.keys[lid as usize]))
            .collect()
    }

    /// Search with range pre-filter: nodes outside [lo, hi] are not added to
    /// the result heap, but their neighborhoods are still traversed
    /// (ACORN-style predicate-agnostic expansion).
    pub fn search_range_prefilter(
        &self, q: &[f32], ef: usize, lo: f32, hi: f32,
    ) -> Vec<(f32, u32, f32)> {
        if self.vectors.is_empty() { return Vec::new(); }
        let mut visited: Vec<bool> = vec![false; self.vectors.len()];
        let start = self.entry as usize;
        visited[start] = true;
        let d0 = l2_sq(q, &self.vectors[start]);
        let mut candidates: BinaryHeap<Reverse<OrdF32>> = BinaryHeap::new();
        let mut results: BinaryHeap<OrdF32> = BinaryHeap::new();
        candidates.push(Reverse(OrdF32(d0, start as u32)));
        if (lo..=hi).contains(&self.keys[start]) {
            results.push(OrdF32(d0, start as u32));
        }
        while let Some(Reverse(c)) = candidates.pop() {
            let worst = results.peek().map(|x| x.0).unwrap_or(f32::INFINITY);
            if results.len() >= ef && c.0 > worst { break; }
            for &nb in &self.neighbors[c.1 as usize] {
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let d = l2_sq(q, &self.vectors[nb as usize]);
                candidates.push(Reverse(OrdF32(d, nb)));
                if (lo..=hi).contains(&self.keys[nb as usize]) {
                    let worst2 = results.peek().map(|x| x.0).unwrap_or(f32::INFINITY);
                    if results.len() < ef || d < worst2 {
                        results.push(OrdF32(d, nb));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }
        let mut out: Vec<(f32, u32, f32)> = results.into_iter()
            .map(|x| (x.0, self.global_ids[x.1 as usize], self.keys[x.1 as usize]))
            .collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }

    pub fn memory_bytes(&self) -> usize {
        let edges: usize = self.neighbors.iter().map(|v| v.capacity()).sum::<usize>() * 4;
        let vecs: usize = self.vectors.iter().map(|v| v.capacity()).sum::<usize>() * 4;
        let meta = self.global_ids.capacity() * 4 + self.keys.capacity() * 4;
        edges + vecs + meta
    }
}
