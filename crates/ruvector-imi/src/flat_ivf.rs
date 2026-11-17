//! Flat IVF baseline: single k-means, search by probing `nprobe`
//! nearest centroids and scanning their postings end-to-end.

use crate::data::Rng;
use crate::kmeans::{kmeans, l2_sq};
use crate::AnnIndex;

pub struct FlatIvf {
    pub d: usize,
    pub k: usize,
    pub centroids: Vec<f32>,
    pub postings: Vec<Vec<u32>>,
    pub vectors: Vec<f32>,
}

impl FlatIvf {
    pub fn build(data: Vec<f32>, d: usize, k: usize, kmeans_iters: usize, seed: u64) -> Self {
        let n = data.len() / d;
        let mut rng = Rng::new(seed);
        let (centroids, assign) = kmeans(&data, d, k, kmeans_iters, &mut rng);
        let mut postings: Vec<Vec<u32>> = (0..k).map(|_| Vec::new()).collect();
        for i in 0..n {
            postings[assign[i] as usize].push(i as u32);
        }
        FlatIvf {
            d,
            k,
            centroids,
            postings,
            vectors: data,
        }
    }
}

impl AnnIndex for FlatIvf {
    fn name(&self) -> &str {
        "flat-ivf"
    }

    fn search(&self, query: &[f32], probe: usize, k: usize) -> Vec<(u32, f32)> {
        // Rank centroids by L2.
        let mut c_scores: Vec<(usize, f32)> = (0..self.k)
            .map(|c| (c, l2_sq(query, &self.centroids[c * self.d..(c + 1) * self.d])))
            .collect();
        c_scores.select_nth_unstable_by(probe.min(self.k) - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });
        c_scores.truncate(probe.min(self.k));

        // Candidate scan.
        let mut heap: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (c, _) in &c_scores {
            for &id in &self.postings[*c] {
                let sq = l2_sq(query, &self.vectors[id as usize * self.d..(id as usize + 1) * self.d]);
                push_top_k(&mut heap, (id, sq), k);
            }
        }
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        heap
    }

    fn memory_bytes(&self) -> usize {
        self.centroids.len() * 4
            + self.postings.iter().map(|p| p.len() * 4 + 24).sum::<usize>()
            + self.vectors.len() * 4
    }
}

#[inline]
pub(crate) fn push_top_k(heap: &mut Vec<(u32, f32)>, item: (u32, f32), k: usize) {
    if heap.len() < k {
        heap.push(item);
        // Maintain heap-ordered by decreasing score at index 0 (max-heap).
        bubble_up_max(heap);
    } else if item.1 < heap[0].1 {
        heap[0] = item;
        sift_down_max(heap);
    }
}

fn bubble_up_max(h: &mut [(u32, f32)]) {
    let mut i = h.len() - 1;
    while i > 0 {
        let p = (i - 1) / 2;
        if h[i].1 > h[p].1 {
            h.swap(i, p);
            i = p;
        } else {
            break;
        }
    }
}

fn sift_down_max(h: &mut [(u32, f32)]) {
    let n = h.len();
    let mut i = 0;
    loop {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        let mut largest = i;
        if l < n && h[l].1 > h[largest].1 {
            largest = l;
        }
        if r < n && h[r].1 > h[largest].1 {
            largest = r;
        }
        if largest == i {
            break;
        }
        h.swap(i, largest);
        i = largest;
    }
}
