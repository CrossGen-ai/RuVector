//! Baseline indices: flat brute force and unpruned IVF.

use crate::{kmeans, l2, sq_l2, AnnIndex, Neighbor, SearchStats};
use std::collections::BinaryHeap;

/// Flat brute-force index — every query scans every vector.
pub struct FlatIndex {
    pub data: Vec<Vec<f32>>,
}

impl FlatIndex {
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        Self { data }
    }
}

impl AnnIndex for FlatIndex {
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
        for (i, x) in self.data.iter().enumerate() {
            let d = sq_l2(q, x).sqrt();
            if heap.len() < k {
                heap.push(Neighbor { id: i as u32, dist: d });
            } else if let Some(top) = heap.peek() {
                if d < top.dist {
                    heap.pop();
                    heap.push(Neighbor { id: i as u32, dist: d });
                }
            }
        }
        let mut out: Vec<Neighbor> = heap.into_sorted_vec();
        out.truncate(k);
        out
    }

    fn estimated_bytes(&self) -> usize {
        let dim = self.data.first().map(|v| v.len()).unwrap_or(0);
        self.data.len() * dim * std::mem::size_of::<f32>()
    }

    fn name(&self) -> &'static str {
        "flat"
    }
}

/// Plain IVF: each point belongs to its nearest centroid; queries scan
/// every point in the top `n_probe` lists. No pruning.
pub struct PlainIvfIndex {
    pub centroids: Vec<Vec<f32>>,
    /// For each cluster: the original point ids in that list.
    pub posting_ids: Vec<Vec<u32>>,
    /// For each cluster: the actual vectors (kept contiguous to mirror
    /// what a real IVF would do).
    pub posting_vecs: Vec<Vec<Vec<f32>>>,
    pub n_probe: usize,
}

impl PlainIvfIndex {
    pub fn build(
        data: Vec<Vec<f32>>,
        n_clusters: usize,
        n_probe: usize,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        let centroids = kmeans(&data, n_clusters, kmeans_iters, seed);
        let mut posting_ids: Vec<Vec<u32>> = vec![Vec::new(); n_clusters];
        let mut posting_vecs: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_clusters];
        for (i, x) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut bd = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d = sq_l2(x, c);
                if d < bd {
                    bd = d;
                    best = j;
                }
            }
            posting_ids[best].push(i as u32);
            posting_vecs[best].push(x.clone());
        }
        Self {
            centroids,
            posting_ids,
            posting_vecs,
            n_probe,
        }
    }

    pub fn probe_clusters(&self, q: &[f32]) -> Vec<(usize, f32)> {
        let mut all: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(j, c)| (j, l2(q, c)))
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all.truncate(self.n_probe);
        all
    }
}

impl AnnIndex for PlainIvfIndex {
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor> {
        self.search_with_stats(q, k).0
    }

    fn search_with_stats(&self, q: &[f32], k: usize) -> (Vec<Neighbor>, SearchStats) {
        let mut stats = SearchStats::default();
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
        for (cluster, _) in self.probe_clusters(q) {
            for (slot, id) in self.posting_ids[cluster].iter().enumerate() {
                let x = &self.posting_vecs[cluster][slot];
                stats.considered += 1;
                stats.full_dist += 1;
                let d = l2(q, x);
                if heap.len() < k {
                    heap.push(Neighbor { id: *id, dist: d });
                } else if let Some(top) = heap.peek() {
                    if d < top.dist {
                        heap.pop();
                        heap.push(Neighbor { id: *id, dist: d });
                    }
                }
            }
        }
        let mut out: Vec<Neighbor> = heap.into_sorted_vec();
        out.truncate(k);
        (out, stats)
    }

    fn estimated_bytes(&self) -> usize {
        let f32sz = std::mem::size_of::<f32>();
        let dim = self.centroids.first().map(|v| v.len()).unwrap_or(0);
        let cent = self.centroids.len() * dim * f32sz;
        let vecs: usize = self
            .posting_vecs
            .iter()
            .map(|p| p.len() * dim * f32sz)
            .sum();
        let ids: usize = self
            .posting_ids
            .iter()
            .map(|p| p.len() * std::mem::size_of::<u32>())
            .sum();
        cent + vecs + ids
    }

    fn name(&self) -> &'static str {
        "ivf-plain"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::make_clustered;

    #[test]
    fn flat_finds_exact_neighbor() {
        let data = make_clustered(500, 8, 5, 0.1, 7);
        let idx = FlatIndex::new(data.clone());
        let q = data[123].clone();
        let neighbors = idx.search(&q, 1);
        assert_eq!(neighbors[0].id, 123);
        assert!(neighbors[0].dist < 1e-5);
    }

    #[test]
    fn ivf_matches_flat_top1_when_probing_all() {
        let data = make_clustered(800, 12, 8, 0.1, 11);
        let flat = FlatIndex::new(data.clone());
        let ivf = PlainIvfIndex::build(data.clone(), 8, 8, 20, 13);
        let q = data[200].clone();
        let a = flat.search(&q, 5);
        let b = ivf.search(&q, 5);
        // Top-1 must match when we probe every cluster.
        assert_eq!(a[0].id, b[0].id);
    }
}
