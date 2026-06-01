//! ruvector-tribase — Triangle-inequality pruning for graph-based ANN.
//!
//! Inspired by Tribase (SIGMOD 2024): use precomputed landmark distances and the
//! triangle inequality `|d(q, L) - d(x, L)| <= d(q, x)` to skip exact distance
//! computations during graph beam search.
//!
//! Three pluggable backends:
//!   * [`BaselineSearcher`] — beam search with no pruning (reference).
//!   * [`TribaseSearcher`]  — one landmark per node (the medoid).
//!   * [`MultiLandmarkSearcher`] — k landmarks per query, max-bound across them.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

pub mod graph;
pub mod metric;
pub mod stats;

pub use graph::FlatGraph;
pub use metric::{l2_sq, dot};
pub use stats::SearchStats;

/// Anything that can answer "what are the approximate top-k neighbors of `query`?".
pub trait AnnSearcher: Send + Sync {
    /// Returns (id, distance) pairs sorted by ascending distance.
    fn search(&self, query: &[f32], k: usize, ef: usize, stats: &mut SearchStats) -> Vec<(u32, f32)>;
    fn name(&self) -> &'static str;
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct Scored {
    id: u32,
    dist: f32,
}
impl Eq for Scored {}
impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        // For a max-heap of "furthest worst", larger dist = greater.
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

/// Reversed scored — used for the min-heap candidate frontier.
#[derive(Copy, Clone, Debug, PartialEq)]
struct ScoredMin {
    id: u32,
    dist: f32,
}
impl Eq for ScoredMin {}
impl Ord for ScoredMin {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for ScoredMin {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

fn beam_search<F>(
    graph: &FlatGraph,
    entry: u32,
    query: &[f32],
    ef: usize,
    mut prune: F,
    stats: &mut SearchStats,
) -> Vec<(u32, f32)>
where
    F: FnMut(u32, f32) -> Option<f32>,
{
    let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
    let mut frontier: BinaryHeap<ScoredMin> = BinaryHeap::new();
    let mut result: BinaryHeap<Scored> = BinaryHeap::new();

    let entry_vec = graph.vector(entry);
    let d0 = l2_sq(query, entry_vec);
    stats.full_dist += 1;
    visited.insert(entry);
    frontier.push(ScoredMin { id: entry, dist: d0 });
    result.push(Scored { id: entry, dist: d0 });

    while let Some(cur) = frontier.pop() {
        let worst = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
        if cur.dist > worst && result.len() >= ef {
            break;
        }
        for &nb in graph.neighbors(cur.id) {
            if !visited.insert(nb) { continue; }
            stats.visited += 1;

            // Try cheap pruning first.
            let worst_now = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
            if result.len() >= ef {
                if let Some(lower_bound) = prune(nb, worst_now) {
                    if lower_bound > worst_now {
                        stats.pruned += 1;
                        continue;
                    }
                }
            }

            let d = l2_sq(query, graph.vector(nb));
            stats.full_dist += 1;

            if result.len() < ef {
                result.push(Scored { id: nb, dist: d });
                frontier.push(ScoredMin { id: nb, dist: d });
            } else if d < result.peek().unwrap().dist {
                result.pop();
                result.push(Scored { id: nb, dist: d });
                frontier.push(ScoredMin { id: nb, dist: d });
            }
        }
    }

    let mut out: Vec<(u32, f32)> = result.into_iter().map(|s| (s.id, s.dist)).collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    out
}

/// Baseline: no pruning, just plain beam search.
pub struct BaselineSearcher<'a> { pub graph: &'a FlatGraph, pub entry: u32 }
impl<'a> AnnSearcher for BaselineSearcher<'a> {
    fn name(&self) -> &'static str { "baseline" }
    fn search(&self, query: &[f32], k: usize, ef: usize, stats: &mut SearchStats) -> Vec<(u32, f32)> {
        let ef = ef.max(k);
        let mut r = beam_search(self.graph, self.entry, query, ef, |_, _| None, stats);
        r.truncate(k);
        r
    }
}

/// Tribase-1: every node stores its distance to one shared landmark (the medoid).
/// During search we compute `d(q, L)` once, then for each candidate `x` we have
/// `|d(q,L) - d(x,L)| <= d(q,x)`, a free lower bound.
pub struct TribaseSearcher<'a> {
    pub graph: &'a FlatGraph,
    pub entry: u32,
    pub landmark_vec: Vec<f32>,
    /// landmark_dist[node] = sqrt-ish distance to landmark (we use Euclidean, not squared,
    /// because the triangle inequality holds for the true metric, not its square).
    pub landmark_dist: Vec<f32>,
}
impl<'a> TribaseSearcher<'a> {
    pub fn build(graph: &'a FlatGraph, entry: u32) -> Self {
        let landmark_vec = pick_medoid(graph);
        let landmark_dist = (0..graph.len() as u32)
            .map(|i| l2_sq(&landmark_vec, graph.vector(i)).sqrt())
            .collect();
        Self { graph, entry, landmark_vec, landmark_dist }
    }
}
impl<'a> AnnSearcher for TribaseSearcher<'a> {
    fn name(&self) -> &'static str { "tribase-1" }
    fn search(&self, query: &[f32], k: usize, ef: usize, stats: &mut SearchStats) -> Vec<(u32, f32)> {
        let ef = ef.max(k);
        let dq_l = l2_sq(&self.landmark_vec, query).sqrt();
        stats.full_dist += 1;
        let prune = |id: u32, worst_sq: f32| -> Option<f32> {
            let dx_l = self.landmark_dist[id as usize];
            let lb = (dq_l - dx_l).abs();
            // Compare squared bounds against squared worst.
            Some(lb * lb)
        };
        // Note: prune returns squared lower bound (we compare against squared distances).
        let _ = worst_dummy_for_doc(); // keep clippy happy on unused capture below
        let mut r = beam_search(self.graph, self.entry, query, ef, prune, stats);
        r.truncate(k);
        r
    }
}

fn worst_dummy_for_doc() {}

/// Tribase-K: K random landmarks, lower bound = max over landmarks.
pub struct MultiLandmarkSearcher<'a> {
    pub graph: &'a FlatGraph,
    pub entry: u32,
    pub landmarks: Vec<Vec<f32>>,
    /// landmark_dist[k][node] = distance to landmark k.
    pub landmark_dist: Vec<Vec<f32>>,
}
impl<'a> MultiLandmarkSearcher<'a> {
    pub fn build(graph: &'a FlatGraph, entry: u32, k_landmarks: usize, seed: u64) -> Self {
        let landmarks = pick_random_landmarks(graph, k_landmarks, seed);
        let landmark_dist: Vec<Vec<f32>> = landmarks.iter()
            .map(|lv| (0..graph.len() as u32)
                .map(|i| l2_sq(lv, graph.vector(i)).sqrt())
                .collect())
            .collect();
        Self { graph, entry, landmarks, landmark_dist }
    }
}
impl<'a> AnnSearcher for MultiLandmarkSearcher<'a> {
    fn name(&self) -> &'static str { "tribase-k" }
    fn search(&self, query: &[f32], k: usize, ef: usize, stats: &mut SearchStats) -> Vec<(u32, f32)> {
        let ef = ef.max(k);
        let dq: Vec<f32> = self.landmarks.iter()
            .map(|lv| l2_sq(lv, query).sqrt())
            .collect();
        stats.full_dist += self.landmarks.len() as u64;

        let landmark_dist = &self.landmark_dist;
        let prune = |id: u32, _worst_sq: f32| -> Option<f32> {
            let mut best_lb = 0.0f32;
            for (li, dq_l) in dq.iter().enumerate() {
                let dx_l = landmark_dist[li][id as usize];
                let lb = (dq_l - dx_l).abs();
                if lb > best_lb { best_lb = lb; }
            }
            Some(best_lb * best_lb)
        };
        let mut r = beam_search(self.graph, self.entry, query, ef, prune, stats);
        r.truncate(k);
        r
    }
}

fn pick_medoid(graph: &FlatGraph) -> Vec<f32> {
    let d = graph.dim();
    let n = graph.len();
    let mut mean = vec![0.0f32; d];
    for i in 0..n as u32 {
        for (m, x) in mean.iter_mut().zip(graph.vector(i).iter()) {
            *m += *x;
        }
    }
    for m in mean.iter_mut() { *m /= n as f32; }
    // Find nearest node to the mean — that's the medoid.
    let (mut best_id, mut best_d) = (0u32, f32::INFINITY);
    for i in 0..n as u32 {
        let dist = l2_sq(&mean, graph.vector(i));
        if dist < best_d { best_d = dist; best_id = i; }
    }
    graph.vector(best_id).to_vec()
}

fn pick_random_landmarks(graph: &FlatGraph, k: usize, seed: u64) -> Vec<Vec<f32>> {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut ids: Vec<u32> = (0..graph.len() as u32).collect();
    ids.shuffle(&mut rng);
    ids.into_iter().take(k).map(|i| graph.vector(i).to_vec()).collect()
}

/// Brute-force exact search for ground truth (used in tests + recall calc).
pub fn brute_force(graph: &FlatGraph, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = (0..graph.len() as u32)
        .map(|i| (i, l2_sq(query, graph.vector(i))))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    all.truncate(k);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph(n: usize, d: usize, seed: u64) -> FlatGraph {
        FlatGraph::random_knn(n, d, 16, seed)
    }

    #[test]
    fn baseline_finds_self() {
        let g = make_graph(200, 16, 1);
        let entry = 0;
        let searcher = BaselineSearcher { graph: &g, entry };
        let q = g.vector(7).to_vec();
        let mut stats = SearchStats::default();
        let res = searcher.search(&q, 5, 32, &mut stats);
        assert!(res.iter().any(|(id, _)| *id == 7), "should find query as own NN");
    }

    #[test]
    fn tribase_preserves_recall() {
        let g = make_graph(500, 16, 2);
        let base = BaselineSearcher { graph: &g, entry: 0 };
        let tri = TribaseSearcher::build(&g, 0);
        let mut sb = SearchStats::default();
        let mut st = SearchStats::default();
        let q = g.vector(123).to_vec();
        let rb = base.search(&q, 10, 64, &mut sb);
        let rt = tri.search(&q, 10, 64, &mut st);
        // Tribase must never miss anything baseline finds at the same ef.
        let ids_b: HashSet<u32> = rb.iter().map(|(i, _)| *i).collect();
        let ids_t: HashSet<u32> = rt.iter().map(|(i, _)| *i).collect();
        assert_eq!(ids_b, ids_t, "tribase must be lossless vs baseline at same ef");
        assert!(st.pruned >= 1, "tribase should prune at least one candidate");
    }

    #[test]
    fn multi_landmark_prunes_more_than_one() {
        let g = make_graph(500, 16, 3);
        let tri1 = TribaseSearcher::build(&g, 0);
        let trik = MultiLandmarkSearcher::build(&g, 0, 8, 99);
        let q = g.vector(50).to_vec();
        let mut s1 = SearchStats::default();
        let mut sk = SearchStats::default();
        tri1.search(&q, 10, 64, &mut s1);
        trik.search(&q, 10, 64, &mut sk);
        assert!(sk.pruned >= s1.pruned,
            "K landmarks should prune at least as much as 1: k={} vs 1={}", sk.pruned, s1.pruned);
    }
}
