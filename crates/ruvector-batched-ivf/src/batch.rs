use crate::ivf::{l2_sq, IvfIndex};

/// A simple top-k min-heap over (-dist, id) pairs.
///
/// We keep the worst (largest-distance) at the top so a push/replace is
/// a single conditional swap plus sift-down. Correctness is independent
/// of insertion order.
#[derive(Clone, Debug)]
pub struct TopK {
    pub k: usize,
    pub items: Vec<(f32, u32)>, // sorted ascending at the end; max-heap during fill
    pub worst: f32,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self { k, items: Vec::with_capacity(k + 1), worst: f32::INFINITY }
    }

    #[inline]
    pub fn push(&mut self, dist: f32, id: u32) {
        if self.items.len() < self.k {
            self.items.push((dist, id));
            self.sift_up(self.items.len() - 1);
            if self.items.len() == self.k {
                self.worst = self.items[0].0;
            }
        } else if dist < self.worst {
            self.items[0] = (dist, id);
            self.sift_down(0);
            self.worst = self.items[0].0;
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let parent = (i - 1) / 2;
            if self.items[parent].0 < self.items[i].0 {
                self.items.swap(parent, i);
                i = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.items.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut big = i;
            if l < n && self.items[l].0 > self.items[big].0 {
                big = l;
            }
            if r < n && self.items[r].0 > self.items[big].0 {
                big = r;
            }
            if big == i {
                break;
            }
            self.items.swap(i, big);
            i = big;
        }
    }

    pub fn into_sorted(mut self) -> Vec<(f32, u32)> {
        self.items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.items
    }
}

/// A batch of queries. Each query gets its own top-k heap.
pub struct QueryBatch {
    pub dim: usize,
    pub queries: Vec<f32>, // row-major
    pub n: usize,
    pub k: usize,
    pub n_probe: usize,
}

impl QueryBatch {
    pub fn new(queries: Vec<f32>, dim: usize, k: usize, n_probe: usize) -> Self {
        assert!(queries.len() % dim == 0, "query dim mismatch");
        let n = queries.len() / dim;
        Self { dim, queries, n, k, n_probe }
    }

    #[inline]
    pub fn query(&self, q: usize) -> &[f32] {
        &self.queries[q * self.dim..(q + 1) * self.dim]
    }

    /// Compute, for every query, the top `n_probe` centroid ids ordered by
    /// ascending distance. Returned as a `[n][n_probe]` flat `Vec<u32>`.
    pub fn probe(&self, index: &IvfIndex) -> Vec<u32> {
        let mut probes = Vec::with_capacity(self.n * self.n_probe);
        let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(index.n_lists);
        for q in 0..self.n {
            scratch.clear();
            for c in 0..index.n_lists {
                let d = l2_sq(self.query(q), index.centroid(c));
                scratch.push((d, c as u32));
            }
            scratch.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for p in 0..self.n_probe {
                probes.push(scratch[p].1);
            }
        }
        probes
    }
}
