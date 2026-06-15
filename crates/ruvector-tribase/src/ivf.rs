//! Standard IVF index (no pruning beyond nprobe).

use crate::dist::{sq_l2, TopK};
use crate::flat::SearchStats;
use crate::kmeans;

pub struct IvfIndex {
    pub centroids: Vec<Vec<f32>>,
    pub lists: Vec<Vec<u32>>, // list of vector ids per centroid
    pub data: Vec<Vec<f32>>,
    pub dim: usize,
}

impl IvfIndex {
    pub fn build(data: Vec<Vec<f32>>, n_lists: usize, kmeans_iters: usize, seed: u64) -> Self {
        let dim = data[0].len();
        let km = kmeans::fit(&data, n_lists, kmeans_iters, seed);
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_lists];
        for (j, &c) in km.assignments.iter().enumerate() {
            lists[c as usize].push(j as u32);
        }
        Self {
            centroids: km.centroids,
            lists,
            data,
            dim,
        }
    }

    pub fn search(&self, q: &[f32], k: usize, nprobe: usize) -> (Vec<(f32, u32)>, SearchStats) {
        let mut stats = SearchStats::default();
        // Distance to every centroid
        let mut cd: Vec<(f32, usize)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (sq_l2(q, c), i))
            .collect();
        stats.dist_computations += self.centroids.len() as u64;
        cd.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let probe = nprobe.min(cd.len());

        let mut heap = TopK::new(k);
        for &(_, ci) in cd[..probe].iter() {
            let list = &self.lists[ci];
            stats.lists_scanned += 1;
            for &vid in list {
                let d = sq_l2(q, &self.data[vid as usize]);
                heap.push(d, vid);
                stats.dist_computations += 1;
                stats.vectors_scanned += 1;
            }
        }
        (heap.into_sorted(), stats)
    }
}
