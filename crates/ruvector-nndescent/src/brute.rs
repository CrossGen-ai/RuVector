//! Brute-force kNN graph: O(N²/2) distances, exact ground truth.
//!
//! Slow but correct. Serves both as the recall denominator and as the
//! "naive baseline" build-time number the other variants must beat.

use crate::distance::DistanceCounter;
use crate::knn_graph::{KnnGraph, KnnGraphBuilder, KnnNeighbor};

pub struct BruteForceBuilder;

impl KnnGraphBuilder for BruteForceBuilder {
    fn name(&self) -> &'static str {
        "BruteForce"
    }

    fn build(&self, vectors: &[Vec<f32>], k: usize, counter: &DistanceCounter) -> KnnGraph {
        let n = vectors.len();
        let mut g = KnnGraph::new(n, k);

        // Precompute upper triangle of distances; insert into both sides.
        let mut tops: Vec<Vec<KnnNeighbor>> = vec![Vec::with_capacity(k + 1); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d = counter.measure(&vectors[i], &vectors[j]);
                push_k(&mut tops[i], KnnNeighbor { id: j as u32, dist: d }, k);
                push_k(&mut tops[j], KnnNeighbor { id: i as u32, dist: d }, k);
            }
        }
        for (i, t) in tops.into_iter().enumerate() {
            let mut s = t;
            s.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            g.neighbors[i] = s;
        }
        g
    }
}

/// Insert into a small top-k buffer kept sorted by distance ascending.
fn push_k(buf: &mut Vec<KnnNeighbor>, cand: KnnNeighbor, k: usize) {
    if buf.len() < k {
        buf.push(cand);
        // simple insertion to keep sorted
        let mut i = buf.len() - 1;
        while i > 0 && buf[i - 1].dist > buf[i].dist {
            buf.swap(i - 1, i);
            i -= 1;
        }
        return;
    }
    // Buf is full and sorted ascending; worst is at the end.
    if cand.dist >= buf[k - 1].dist {
        return;
    }
    buf[k - 1] = cand;
    let mut i = k - 1;
    while i > 0 && buf[i - 1].dist > buf[i].dist {
        buf.swap(i - 1, i);
        i -= 1;
    }
}

pub(crate) fn push_k_pub(buf: &mut Vec<KnnNeighbor>, cand: KnnNeighbor, k: usize) {
    push_k(buf, cand, k);
}
