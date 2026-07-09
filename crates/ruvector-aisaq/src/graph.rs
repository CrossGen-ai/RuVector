//! Static k-NN graph + beam search.
//!
//! We build a directed graph where each node has out-degree `R`,
//! populated by brute-force nearest neighbours over the training
//! set. This is a Vamana/DiskANN-lite navigation index — enough
//! to expose the AISAQ storage question, without dragging in a
//! full Vamana implementation.
//!
//! The graph is `n * R * 4` bytes and stays in RAM in all
//! variants. That's the AISAQ invariant: the graph is cheap,
//! the codes are expensive, so put the expensive thing on SSD.

use std::collections::BinaryHeap;
use std::cmp::Reverse;

use crate::backends::DistanceBackend;
use crate::l2_sq;

/// A flat, directed k-NN graph: `neighbors[i * r .. (i+1) * r]`
/// are the out-neighbours of node `i`. Empty slots are `u32::MAX`.
pub struct KnnGraph {
    pub n: usize,
    pub r: usize,
    pub neighbors: Vec<u32>,
    pub entry: u32,
}

impl KnnGraph {
    /// Bytes held in RAM by the graph itself (excluding any vector data).
    pub fn ram_bytes(&self) -> usize {
        self.neighbors.len() * std::mem::size_of::<u32>()
    }

    /// Brute-force build: for each node, take the R nearest ground-truth
    /// neighbours from the raw f32 vectors. O(N^2 * D) — fine for the
    /// benchmark's N <= 20k. Production would swap in Vamana's
    /// robust-prune + two-pass build.
    pub fn build_bruteforce(vectors: &[f32], d: usize, r: usize) -> Self {
        let n = vectors.len() / d;
        let mut neighbors = vec![u32::MAX; n * r];
        for i in 0..n {
            let vi = &vectors[i * d..(i + 1) * d];
            // (dist, id) heap of size r, keep smallest r
            let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::with_capacity(r + 1);
            // encode dist as u32 bits (order-preserving for non-negative f32)
            for j in 0..n {
                if i == j { continue; }
                let vj = &vectors[j * d..(j + 1) * d];
                let dist = l2_sq(vi, vj);
                let key = dist.to_bits();
                if heap.len() < r {
                    heap.push((key, j as u32));
                } else if let Some(&(top, _)) = heap.peek() {
                    if key < top {
                        heap.pop();
                        heap.push((key, j as u32));
                    }
                }
            }
            let mut buf: Vec<(u32, u32)> = heap.into_sorted_vec();
            // into_sorted_vec gives ascending on the max-heap's ordering (small first)
            buf.truncate(r);
            for (k, (_, id)) in buf.iter().enumerate() {
                neighbors[i * r + k] = *id;
            }
        }
        // Pick entry as the medoid-ish: node closest to the mean.
        let mut mean = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..d { mean[j] += vectors[i * d + j]; }
        }
        for v in mean.iter_mut() { *v /= n as f32; }
        let mut best = 0u32;
        let mut best_d = f32::INFINITY;
        for i in 0..n {
            let dist = l2_sq(&vectors[i * d..(i + 1) * d], &mean);
            if dist < best_d { best_d = dist; best = i as u32; }
        }
        Self { n, r, neighbors, entry: best }
    }

    #[inline]
    pub fn neighbors_of(&self, id: u32) -> &[u32] {
        &self.neighbors[id as usize * self.r..(id as usize + 1) * self.r]
    }
}

/// Beam-search wrapper. The `backend` decides how distances are
/// computed (raw f32 in RAM, PQ codes in RAM, PQ codes on SSD).
pub struct BeamSearcher<'g, B: DistanceBackend> {
    pub graph: &'g KnnGraph,
    pub backend: B,
    pub beam: usize,
}

impl<'g, B: DistanceBackend> BeamSearcher<'g, B> {
    pub fn new(graph: &'g KnnGraph, backend: B, beam: usize) -> Self {
        Self { graph, backend, beam }
    }

    /// Best-first search. Returns top-k node ids in ascending distance.
    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<u32> {
        self.backend.prepare_query(query);

        // Candidate: (dist_bits, id). Use min-heap by wrapping in Reverse.
        let mut candidates: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        // Results: max-heap of size <= beam.
        let mut results: BinaryHeap<(u32, u32)> = BinaryHeap::new();
        let mut visited = vec![false; self.graph.n];

        let entry = self.graph.entry;
        let d0 = self.backend.dist(entry).to_bits();
        candidates.push(Reverse((d0, entry)));
        results.push((d0, entry));
        visited[entry as usize] = true;

        while let Some(Reverse((d_cand, id))) = candidates.pop() {
            // If the worst in results is closer than best candidate, stop.
            if let Some(&(worst_res, _)) = results.peek() {
                if results.len() >= self.beam && d_cand > worst_res {
                    break;
                }
            }
            for &nb in self.graph.neighbors_of(id) {
                if nb == u32::MAX { continue; }
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let dn = self.backend.dist(nb).to_bits();
                candidates.push(Reverse((dn, nb)));
                if results.len() < self.beam {
                    results.push((dn, nb));
                } else if let Some(&(worst_res, _)) = results.peek() {
                    if dn < worst_res {
                        results.pop();
                        results.push((dn, nb));
                    }
                }
            }
        }

        let mut sorted: Vec<(u32, u32)> = results.into_vec();
        sorted.sort_by_key(|(d, _)| *d);
        sorted.into_iter().take(k).map(|(_, id)| id).collect()
    }
}
