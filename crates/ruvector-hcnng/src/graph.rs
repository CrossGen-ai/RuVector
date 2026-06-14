//! Adjacency-list proximity graph for HCNNG.
//!
//! We store CSR-style flat adjacency for cache-friendly traversal during
//! search. Edges are deduplicated; each node's neighbor list is truncated
//! to `max_degree` nearest neighbors (by distance) once construction is done.

use crate::distance::Distance;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub n: usize,
    /// neighbors[i] = sorted-by-distance neighbor list of node i.
    pub neighbors: Vec<Vec<u32>>,
}

impl Graph {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            neighbors: vec![Vec::new(); n],
        }
    }

    pub fn add_edge_unique(&mut self, u: u32, v: u32) {
        if u == v {
            return;
        }
        let ul = &mut self.neighbors[u as usize];
        if !ul.contains(&v) {
            ul.push(v);
        }
        let vl = &mut self.neighbors[v as usize];
        if !vl.contains(&u) {
            vl.push(u);
        }
    }

    /// Sort each adjacency list by distance to its owner and truncate to `max_degree`.
    pub fn finalize<D: Distance + ?Sized>(
        &mut self,
        vectors: &[Vec<f32>],
        dist: &D,
        max_degree: usize,
    ) {
        for i in 0..self.n {
            // Compute (distance, neighbor) pairs.
            let mut scored: Vec<(f32, u32)> = self.neighbors[i]
                .iter()
                .map(|&j| (dist.d(&vectors[i], &vectors[j as usize]), j))
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            if scored.len() > max_degree {
                scored.truncate(max_degree);
            }
            self.neighbors[i] = scored.into_iter().map(|(_, j)| j).collect();
        }
    }

    pub fn avg_degree(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let total: usize = self.neighbors.iter().map(|v| v.len()).sum();
        total as f64 / self.n as f64
    }

    pub fn max_degree(&self) -> usize {
        self.neighbors.iter().map(|v| v.len()).max().unwrap_or(0)
    }

    pub fn edge_count(&self) -> usize {
        let total: usize = self.neighbors.iter().map(|v| v.len()).sum();
        total / 2
    }
}
