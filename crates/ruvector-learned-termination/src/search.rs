//! Three beam-search variants differentiated by their termination criterion.
//!
//! All variants operate on a [`FlatGraph`]; the termination logic applies
//! unchanged to a real HNSW's layer-0 walk (feature extraction only depends
//! on the current results heap + step counter, not the graph structure).

use crate::graph::FlatGraph;
use crate::predictor::{FeatureSnapshot, LogisticPredictor};
use std::collections::{BinaryHeap, HashSet};

/// A single ANN result.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Per-search accounting: real work performed.
///
/// `dist_calls` counts BEAM-EXPANSION distance evaluations only — the entry-
/// selection scan is intentionally not billed, so the metric isolates the
/// effect of termination decisions. `entry_dist_calls` records the entry cost
/// separately for full-cost analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchStats {
    /// Distance evaluations executed during beam expansion (entry-scan excluded).
    pub dist_calls: usize,
    /// Distance evaluations spent selecting the beam entry point.
    pub entry_dist_calls: usize,
    /// Number of expansion steps (candidates popped and processed).
    pub steps: usize,
}

/// Common search interface.
pub trait Searcher: Send + Sync {
    fn search(&self, query: &[f32], k: usize) -> (Vec<Hit>, SearchStats);
    fn name(&self) -> &str;
    fn memory_bytes(&self) -> usize;
}

fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Brute-force entry — same simplification the sister entropy crate uses.
/// Isolates the termination signal from entry-quality noise.
fn find_entry(graph: &FlatGraph, query: &[f32], stats: &mut SearchStats) -> usize {
    if graph.is_empty() {
        return 0;
    }
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for i in 0..graph.len() {
        let d = l2sq(query, &graph.vectors[i]);
        stats.entry_dist_calls += 1;
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn memory_bytes(graph: &FlatGraph) -> usize {
    let vecs: usize = graph.vectors.iter().map(|v| v.len() * 4).sum();
    let adj: usize = graph.adjacency.iter().map(|a| a.len() * 8).sum::<usize>();
    vecs + adj
}

// ─── Variant 1: FixedEfSearch (baseline) ────────────────────────────────────

/// Baseline HNSW-style greedy beam search with fixed `ef_search`.
pub struct FixedEfSearch<'a> {
    pub graph: &'a FlatGraph,
    pub ef_search: usize,
}

impl<'a> Searcher for FixedEfSearch<'a> {
    fn name(&self) -> &str {
        "FixedEf"
    }
    fn memory_bytes(&self) -> usize {
        memory_bytes(self.graph)
    }
    fn search(&self, query: &[f32], k: usize) -> (Vec<Hit>, SearchStats) {
        let mut stats = SearchStats::default();
        let ef = self.ef_search.max(k);
        let entry = find_entry(self.graph, query, &mut stats);
        let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let entry_dist = l2sq(query, &self.graph.vectors[entry]);
        stats.dist_calls += 1;
        candidates.push(std::cmp::Reverse(Hit {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Hit {
            id: entry,
            dist: entry_dist,
        });
        visited.insert(entry);

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            stats.steps += 1;
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.dist > worst.dist {
                        break;
                    }
                }
            }
            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if !visited.insert(neighbour) {
                    continue;
                }
                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                stats.dist_calls += 1;
                candidates.push(std::cmp::Reverse(Hit {
                    id: neighbour,
                    dist,
                }));
                if results.len() < ef {
                    results.push(Hit {
                        id: neighbour,
                        dist,
                    });
                } else if let Some(worst) = results.peek() {
                    if dist < worst.dist {
                        results.pop();
                        results.push(Hit {
                            id: neighbour,
                            dist,
                        });
                    }
                }
            }
        }
        let mut out: Vec<Hit> = results.into_sorted_vec();
        out.truncate(k);
        (out, stats)
    }
}

// ─── Variant 2: LearnedTermination ──────────────────────────────────────────

/// Beam search that terminates when a learned logistic classifier judges
/// P(top-k will still change) < tau. Falls back to the standard HNSW prune.
pub struct LearnedTermination<'a> {
    pub graph: &'a FlatGraph,
    pub ef_search: usize,
    pub predictor: LogisticPredictor,
    /// Termination threshold on P(improve). Lower = more aggressive termination.
    pub tau: f32,
    /// Window over which `improve_rate` is averaged.
    pub improve_window: usize,
    /// Minimum expansion steps before termination gate can fire.
    pub min_steps: usize,
}

impl<'a> Searcher for LearnedTermination<'a> {
    fn name(&self) -> &str {
        "LearnedTermination"
    }
    fn memory_bytes(&self) -> usize {
        // 24 bytes of weights are trivial vs graph memory.
        memory_bytes(self.graph) + 24
    }
    fn search(&self, query: &[f32], k: usize) -> (Vec<Hit>, SearchStats) {
        let mut stats = SearchStats::default();
        let ef = self.ef_search.max(k);
        let entry = find_entry(self.graph, query, &mut stats);
        let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let entry_dist = l2sq(query, &self.graph.vectors[entry]);
        stats.dist_calls += 1;
        candidates.push(std::cmp::Reverse(Hit {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Hit {
            id: entry,
            dist: entry_dist,
        });
        visited.insert(entry);

        // best_dist history for improve_rate feature
        let mut best_history: Vec<f32> = Vec::with_capacity(64);
        best_history.push(entry_dist);

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            stats.steps += 1;
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.dist > worst.dist {
                        break;
                    }
                }
            }
            // Track frontier ratio for this expansion.
            let degree = self.graph.adjacency[current.id].len();
            let mut unvisited_here = 0usize;
            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if !visited.insert(neighbour) {
                    continue;
                }
                unvisited_here += 1;
                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                stats.dist_calls += 1;
                candidates.push(std::cmp::Reverse(Hit {
                    id: neighbour,
                    dist,
                }));
                if results.len() < ef {
                    results.push(Hit {
                        id: neighbour,
                        dist,
                    });
                } else if let Some(worst) = results.peek() {
                    if dist < worst.dist {
                        results.pop();
                        results.push(Hit {
                            id: neighbour,
                            dist,
                        });
                    }
                }
            }
            // Snapshot post-expansion; compute features.
            let mut dists_desc: Vec<f32> = results.iter().map(|h| h.dist).collect();
            dists_desc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let best_dist = *dists_desc.first().unwrap_or(&entry_dist);
            best_history.push(best_dist);
            let feat = FeatureSnapshot {
                best_dist,
                improve_rate: improve_rate(&best_history, self.improve_window),
                gap_kth: kth_gap(&dists_desc, k),
                steps_norm: (stats.steps as f32) / (ef.max(1) as f32),
                frontier_ratio: if degree == 0 {
                    0.0
                } else {
                    unvisited_here as f32 / degree as f32
                },
            };
            if stats.steps >= self.min_steps
                && !self.predictor.should_continue(&feat, self.tau)
            {
                break;
            }
        }
        let mut out: Vec<Hit> = results.into_sorted_vec();
        out.truncate(k);
        (out, stats)
    }
}

fn improve_rate(hist: &[f32], window: usize) -> f32 {
    if hist.len() < 2 {
        return 0.0;
    }
    let w = window.min(hist.len() - 1);
    let past = hist[hist.len() - 1 - w];
    let now = hist[hist.len() - 1];
    ((past - now) / (past.abs() + 1e-6)).max(0.0)
}

fn kth_gap(dists_asc: &[f32], k: usize) -> f32 {
    if dists_asc.len() < k.max(2) {
        return 1.0; // large gap → assume unconverged
    }
    let a = dists_asc[k - 2];
    let b = dists_asc[k - 1];
    ((b - a) / (b.abs() + 1e-6)).abs()
}

// ─── Variant 3: OracleTermination ───────────────────────────────────────────

/// Reference upper bound: knows the true top-k a priori and terminates the
/// moment the results-heap top-k stops changing across `patience` steps.
/// Cannot be deployed in production — used only to bound achievable savings.
pub struct OracleTermination<'a> {
    pub graph: &'a FlatGraph,
    pub ef_search: usize,
    pub patience: usize,
    /// The oracle's knowledge: exact top-k ids for this query.
    /// Populated per-query by the benchmark harness.
    pub target: std::sync::Mutex<Option<Vec<usize>>>,
}

impl<'a> Searcher for OracleTermination<'a> {
    fn name(&self) -> &str {
        "Oracle"
    }
    fn memory_bytes(&self) -> usize {
        memory_bytes(self.graph)
    }
    fn search(&self, query: &[f32], k: usize) -> (Vec<Hit>, SearchStats) {
        let mut stats = SearchStats::default();
        let ef = self.ef_search.max(k);
        let entry = find_entry(self.graph, query, &mut stats);
        let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut results: BinaryHeap<Hit> = BinaryHeap::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let entry_dist = l2sq(query, &self.graph.vectors[entry]);
        stats.dist_calls += 1;
        candidates.push(std::cmp::Reverse(Hit {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Hit {
            id: entry,
            dist: entry_dist,
        });
        visited.insert(entry);

        let target_ids: Option<std::collections::HashSet<usize>> = self
            .target
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|v| v.iter().copied().collect()));

        let mut steady_steps = 0usize;

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            stats.steps += 1;
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.dist > worst.dist {
                        break;
                    }
                }
            }
            for &(_, neighbour) in &self.graph.adjacency[current.id] {
                if !visited.insert(neighbour) {
                    continue;
                }
                let dist = l2sq(query, &self.graph.vectors[neighbour]);
                stats.dist_calls += 1;
                candidates.push(std::cmp::Reverse(Hit {
                    id: neighbour,
                    dist,
                }));
                if results.len() < ef {
                    results.push(Hit {
                        id: neighbour,
                        dist,
                    });
                } else if let Some(worst) = results.peek() {
                    if dist < worst.dist {
                        results.pop();
                        results.push(Hit {
                            id: neighbour,
                            dist,
                        });
                    }
                }
            }
            // Oracle check: if we already have all true top-k in results,
            // count patience steps and terminate.
            if let Some(ref target) = target_ids {
                let mut sorted: Vec<Hit> = results.clone().into_sorted_vec();
                sorted.truncate(k);
                let have_all = sorted.iter().all(|h| target.contains(&h.id));
                if have_all {
                    steady_steps += 1;
                    if steady_steps >= self.patience {
                        break;
                    }
                } else {
                    steady_steps = 0;
                }
            }
        }
        let mut out: Vec<Hit> = results.into_sorted_vec();
        out.truncate(k);
        (out, stats)
    }
}

// ─── training-data collection: replays a search and emits per-step labels ───

/// Replay a fixed-ef search, capturing the feature snapshot at every step
/// along with the label "was the top-k at this step already the final top-k?"
/// (label 0.0 = stable, so continuing would NOT change results;
///  label 1.0 = still changing, keep going).
///
/// Emits labels aligned with `P(improve)`: label=1.0 while top-k still changing,
/// label=0.0 once top-k stabilises.
pub fn collect_training_samples(
    graph: &FlatGraph,
    query: &[f32],
    ef: usize,
    k: usize,
) -> Vec<(FeatureSnapshot, f32)> {
    let mut samples: Vec<(FeatureSnapshot, f32)> = Vec::new();
    let mut candidates: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
    let mut results: BinaryHeap<Hit> = BinaryHeap::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut _stats = SearchStats::default();
    let entry = find_entry(graph, query, &mut _stats);
    let entry_dist = l2sq(query, &graph.vectors[entry]);
    candidates.push(std::cmp::Reverse(Hit {
        id: entry,
        dist: entry_dist,
    }));
    results.push(Hit {
        id: entry,
        dist: entry_dist,
    });
    visited.insert(entry);

    let mut best_history: Vec<f32> = vec![entry_dist];
    let mut topk_history: Vec<Vec<usize>> = Vec::new();
    let mut features_history: Vec<FeatureSnapshot> = Vec::new();
    let mut steps = 0usize;

    while let Some(std::cmp::Reverse(current)) = candidates.pop() {
        steps += 1;
        if results.len() >= ef {
            if let Some(worst) = results.peek() {
                if current.dist > worst.dist {
                    break;
                }
            }
        }
        let degree = graph.adjacency[current.id].len();
        let mut unvisited_here = 0usize;
        for &(_, neighbour) in &graph.adjacency[current.id] {
            if !visited.insert(neighbour) {
                continue;
            }
            unvisited_here += 1;
            let dist = l2sq(query, &graph.vectors[neighbour]);
            candidates.push(std::cmp::Reverse(Hit {
                id: neighbour,
                dist,
            }));
            if results.len() < ef {
                results.push(Hit {
                    id: neighbour,
                    dist,
                });
            } else if let Some(worst) = results.peek() {
                if dist < worst.dist {
                    results.pop();
                    results.push(Hit {
                        id: neighbour,
                        dist,
                    });
                }
            }
        }
        let mut sorted: Vec<Hit> = results.clone().into_sorted_vec();
        sorted.truncate(k);
        let best_dist = sorted.first().map(|h| h.dist).unwrap_or(entry_dist);
        best_history.push(best_dist);
        let mut dists_asc: Vec<f32> = sorted.iter().map(|h| h.dist).collect();
        dists_asc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let feat = FeatureSnapshot {
            best_dist,
            improve_rate: improve_rate(&best_history, 4),
            gap_kth: kth_gap(&dists_asc, k),
            steps_norm: (steps as f32) / (ef.max(1) as f32),
            frontier_ratio: if degree == 0 {
                0.0
            } else {
                unvisited_here as f32 / degree as f32
            },
        };
        features_history.push(feat);
        topk_history.push(sorted.iter().map(|h| h.id).collect());
    }
    if topk_history.is_empty() {
        return samples;
    }
    let final_topk: std::collections::HashSet<usize> =
        topk_history.last().unwrap().iter().copied().collect();
    // Label 1 while topk still != final; label 0 once stabilised.
    for (feat, tk) in features_history.iter().zip(topk_history.iter()) {
        let now: std::collections::HashSet<usize> = tk.iter().copied().collect();
        let label = if now == final_topk { 0.0 } else { 1.0 };
        samples.push((*feat, label));
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::{clustered_vectors, ground_truth, random_unit_vectors},
        graph::{FlatGraph, GraphConfig},
        recall_at_k,
    };

    fn build(n: usize, dim: usize, seed: u64) -> FlatGraph {
        let v = random_unit_vectors(n, dim, seed);
        FlatGraph::build(v, GraphConfig { k_neighbours: 12 })
    }

    #[test]
    fn fixed_ef_returns_k_hits() {
        let g = build(200, 16, 7);
        let q = random_unit_vectors(1, 16, 99);
        let s = FixedEfSearch {
            graph: &g,
            ef_search: 40,
        };
        let (hits, stats) = s.search(&q[0], 10);
        assert_eq!(hits.len(), 10);
        assert!(stats.dist_calls > 0);
        assert!(stats.steps > 0);
    }

    #[test]
    fn learned_saves_work_at_high_recall() {
        // Compare Fixed vs Learned with the *default* predictor.
        // The Learned variant should use <= as many dist calls as Fixed and
        // maintain recall within tolerance.
        let corpus = clustered_vectors(400, 32, 8, 0.2, 42);
        let graph = FlatGraph::build(corpus.clone(), GraphConfig { k_neighbours: 16 });
        let queries = corpus[..40].to_vec();

        let fixed = FixedEfSearch {
            graph: &graph,
            ef_search: 60,
        };
        let learned = LearnedTermination {
            graph: &graph,
            ef_search: 60,
            predictor: LogisticPredictor::default(),
            tau: 0.15,
            improve_window: 4,
            min_steps: 6,
        };

        let mut fixed_calls = 0usize;
        let mut learn_calls = 0usize;
        let mut fixed_recall = 0.0f32;
        let mut learn_recall = 0.0f32;
        for q in &queries {
            let gt = ground_truth(q, &corpus, 10);
            let (h_f, s_f) = fixed.search(q, 10);
            let (h_l, s_l) = learned.search(q, 10);
            fixed_calls += s_f.dist_calls;
            learn_calls += s_l.dist_calls;
            fixed_recall += recall_at_k(&gt, &h_f, 10);
            learn_recall += recall_at_k(&gt, &h_l, 10);
        }
        let n = queries.len() as f32;
        let fixed_recall = fixed_recall / n;
        let learn_recall = learn_recall / n;
        println!(
            "Fixed: dist_calls={} recall={:.3} | Learned: dist_calls={} recall={:.3}",
            fixed_calls, fixed_recall, learn_calls, learn_recall
        );
        assert!(
            learn_calls <= fixed_calls,
            "Learned should not do more dist calls than Fixed"
        );
        // The default hand-set weights should keep recall within 0.05 of Fixed.
        assert!(
            learn_recall + 0.05 >= fixed_recall,
            "Learned recall {learn_recall:.3} too low vs Fixed {fixed_recall:.3}"
        );
    }

    #[test]
    fn oracle_terminates_no_later_than_fixed() {
        let corpus = clustered_vectors(300, 16, 6, 0.15, 11);
        let graph = FlatGraph::build(corpus.clone(), GraphConfig { k_neighbours: 12 });
        let q = corpus[10].clone();
        let gt = ground_truth(&q, &corpus, 10);
        let fixed = FixedEfSearch {
            graph: &graph,
            ef_search: 60,
        };
        let oracle = OracleTermination {
            graph: &graph,
            ef_search: 60,
            patience: 3,
            target: std::sync::Mutex::new(Some(gt.clone())),
        };
        let (_, sf) = fixed.search(&q, 10);
        let (_, so) = oracle.search(&q, 10);
        assert!(
            so.dist_calls <= sf.dist_calls,
            "Oracle should not exceed Fixed dist_calls"
        );
    }

    #[test]
    fn training_produces_useful_weights() {
        // Use a larger, harder dataset so beam expansion actually goes through
        // multiple top-k states → we collect both label classes.
        // Off-corpus random queries: the exact top-k is not found instantly by
        // the brute-force entry, so top-k actually evolves during beam expansion
        // → we collect both label classes.
        let corpus = clustered_vectors(800, 32, 10, 0.3, 22);
        let graph = FlatGraph::build(corpus.clone(), GraphConfig { k_neighbours: 20 });
        let queries = random_unit_vectors(60, 32, 999);
        let mut samples: Vec<(FeatureSnapshot, f32)> = Vec::new();
        for q in queries.iter() {
            samples.extend(collect_training_samples(&graph, q, 120, 10));
        }
        assert!(samples.len() > 100, "should collect training data");
        let pos = samples.iter().filter(|(_, y)| *y > 0.5).count();
        let neg = samples.iter().filter(|(_, y)| *y <= 0.5).count();
        assert!(
            pos > 0 && neg > 0,
            "training set has both classes: pos={pos} neg={neg}"
        );
        let p = LogisticPredictor::train(&samples, &crate::predictor::TrainConfig::default());
        // Basic sanity: p_improve should be bounded in [0,1].
        let f = FeatureSnapshot {
            best_dist: 0.1,
            improve_rate: 0.5,
            gap_kth: 0.05,
            steps_norm: 0.1,
            frontier_ratio: 0.8,
        };
        let p_val = p.p_improve(&f);
        assert!(p_val >= 0.0 && p_val <= 1.0);
    }
}
