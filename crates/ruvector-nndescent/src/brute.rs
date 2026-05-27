//! Exact brute-force k-NN graph baseline — O(N²) distance calls.
//!
//! Used both for ground-truth in recall measurement and as the "no-go faster
//! than this with exact answer" reference point in benchmarks.

use crate::{heap::BoundedMaxHeap, BuildReport, KnnGraph, KnnGraphBuilder, Metric};
use std::time::Instant;

pub struct BruteForce<M: Metric> {
    pub metric: M,
}

impl<M: Metric> BruteForce<M> {
    pub fn new(metric: M) -> Self { Self { metric } }
}

impl<M: Metric> KnnGraphBuilder for BruteForce<M> {
    fn build(&mut self, data: &[Vec<f32>], k: usize) -> BuildReport {
        let n = data.len();
        let t0 = Instant::now();
        let mut graph: KnnGraph = Vec::with_capacity(n);
        let mut calls: u64 = 0;

        // Distance is symmetric for L2/cosine etc., but we don't enforce it
        // here so unsymmetric metrics still produce correct ground-truth.
        for i in 0..n {
            let mut heap = BoundedMaxHeap::new(k);
            for j in 0..n {
                if i == j { continue; }
                let d = self.metric.dist(&data[i], &data[j]);
                calls += 1;
                heap.push(j as u32, d, false);
            }
            graph.push(heap.into_sorted());
        }

        BuildReport {
            graph,
            elapsed: t0.elapsed(),
            distance_calls: calls,
            iterations: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::L2;

    #[test]
    fn brute_neighbours_are_sorted_and_exclude_self() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![3.0, 0.0],
            vec![6.0, 0.0],
        ];
        let mut b = BruteForce::new(L2);
        let r = b.build(&data, 2);
        // Node 0's two nearest are 1 (d=1) then 2 (d=9).
        assert_eq!(r.graph[0].len(), 2);
        assert_eq!(r.graph[0][0].id, 1);
        assert_eq!(r.graph[0][1].id, 2);
        assert!(r.graph[0][0].dist <= r.graph[0][1].dist);
        // Self never appears.
        for (i, nbrs) in r.graph.iter().enumerate() {
            for n in nbrs {
                assert_ne!(n.id as usize, i);
            }
        }
    }
}
