//! Greedy graph search identical across layouts. We report both wall time
//! *and* recall — a good layout must not change recall.

use crate::graph::{l2sq, MiniHnsw, Ordered};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;

#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct SearchStats {
    pub layout: String,
    pub queries: usize,
    pub k: usize,
    pub ef: usize,
    pub total_ns: u128,
    pub qps: f64,
    pub recall_at_k: f64,
    pub visited_avg: f64,
}

/// Standard beam-search on the proximity graph. Returns top-k ids.
pub fn search_greedy(g: &MiniHnsw, query: &[f32], ef: usize, k: usize) -> (Vec<u32>, usize) {
    let n = g.len();
    let mut visited = vec![false; n];
    let mut best: BinaryHeap<Ordered> = BinaryHeap::new(); // max-heap, size = ef
    let mut frontier: BinaryHeap<std::cmp::Reverse<Ordered>> = BinaryHeap::new(); // min-heap
    let start = g.entry;
    let d0 = l2sq(query, g.vector(start));
    visited[start as usize] = true;
    best.push(Ordered(d0, start));
    frontier.push(std::cmp::Reverse(Ordered(d0, start)));
    let mut visits = 1usize;
    while let Some(std::cmp::Reverse(Ordered(d, id))) = frontier.pop() {
        if let Some(worst) = best.peek() {
            if d > worst.0 && best.len() >= ef {
                break;
            }
        }
        for &nb in &g.neighbors[id as usize] {
            if visited[nb as usize] {
                continue;
            }
            visited[nb as usize] = true;
            visits += 1;
            let dd = l2sq(query, g.vector(nb));
            if best.len() < ef {
                best.push(Ordered(dd, nb));
                frontier.push(std::cmp::Reverse(Ordered(dd, nb)));
            } else if dd < best.peek().unwrap().0 {
                best.pop();
                best.push(Ordered(dd, nb));
                frontier.push(std::cmp::Reverse(Ordered(dd, nb)));
            }
        }
    }
    let mut out: Vec<Ordered> = best.into_iter().collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (out.into_iter().take(k).map(|Ordered(_, id)| id).collect(), visits)
}

/// Compute exact (brute-force) top-k on the SAME vector data — used as the
/// recall ground truth. Layout doesn't matter for exact search.
pub fn exact_topk(g: &MiniHnsw, query: &[f32], k: usize) -> Vec<u32> {
    let mut all: Vec<Ordered> = (0..g.len() as u32)
        .map(|i| Ordered(l2sq(query, g.vector(i)), i))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    all.into_iter().take(k).map(|Ordered(_, id)| id).collect()
}

/// End-to-end bench: run `queries.len()` searches, return timing + recall.
/// Because layouts permute IDs, we must translate ground-truth IDs (which
/// refer to the *original* vectors) via `id_map`, where `id_map[old] = new`.
pub fn bench_layout(
    g: &MiniHnsw,
    queries: &[Vec<f32>],
    truth: &[Vec<u32>],
    id_map: &[u32],
    ef: usize,
    k: usize,
    layout_name: &str,
) -> SearchStats {
    let mut correct = 0usize;
    let mut total_visits = 0usize;
    let t0 = std::time::Instant::now();
    for (q, gt) in queries.iter().zip(truth.iter()) {
        let (res, v) = search_greedy(g, q, ef, k);
        total_visits += v;
        // Translate ground-truth ids into this graph's namespace.
        let gt_set: std::collections::HashSet<u32> =
            gt.iter().map(|&id| id_map[id as usize]).collect();
        for r in &res {
            if gt_set.contains(r) {
                correct += 1;
            }
        }
    }
    let el = t0.elapsed().as_nanos();
    let queries_n = queries.len();
    let qps = queries_n as f64 / (el as f64 / 1e9);
    SearchStats {
        layout: layout_name.to_string(),
        queries: queries_n,
        k,
        ef,
        total_ns: el,
        qps,
        recall_at_k: correct as f64 / (queries_n * k) as f64,
        visited_avg: total_visits as f64 / queries_n as f64,
    }
}
