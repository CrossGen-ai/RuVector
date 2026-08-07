//! Minimal, self-contained HNSW-style ANN index sufficient for studying the
//! `ef_search` parameter. Single graph layer (equivalent to HNSW layer 0),
//! greedy entry + best-first candidate expansion. Distance counts are exposed
//! so the benchmark can report work per query.
//!
//! This is intentionally compact (< 400 lines) so the whole crate is
//! self-contained: no `hnsw_rs` dependency, no unsafe, deterministic.

use crate::dataset::{l2_sq, Rng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Distance counted so benchmarks can report exact work.
#[derive(Debug, Default, Clone, Copy)]
pub struct ProbeStats {
    pub dists: u64,
    pub d_entry: f32,
    pub d1: f32,
    pub d2: f32,
    pub first_hop_mean: f32,
}

#[derive(Copy, Clone, PartialEq)]
struct MinItem {
    dist: f32,
    id: u32,
}
impl Eq for MinItem {}
impl Ord for MinItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse compare (smaller dist = larger priority).
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MinItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

#[derive(Copy, Clone, PartialEq)]
struct MaxItem {
    dist: f32,
    id: u32,
}
impl Eq for MaxItem {}
impl Ord for MaxItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MaxItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

pub struct Hnsw {
    vectors: Vec<Vec<f32>>,
    neighbors: Vec<Vec<u32>>, // adjacency
    m: usize,                 // neighbors per node
    entries: Vec<u32>,        // multiple entry points for connectivity
}

impl Hnsw {
    /// Build a graph by inserting vectors one at a time using best-first
    /// search with `ef_construction`. Bidirectional links, pruned to `m`.
    pub fn build(vectors: Vec<Vec<f32>>, m: usize, ef_construction: usize, seed: u64) -> Self {
        assert!(!vectors.is_empty());
        let n = vectors.len();
        let mut neighbors: Vec<Vec<u32>> = (0..n).map(|_| Vec::with_capacity(m * 2)).collect();
        let mut rng = Rng::new(seed);
        let _ = rng.next_u64();

        // Insertion order: random permutation for balance.
        let mut order: Vec<u32> = (0..n as u32).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }

        let entry = order[0];
        for &idx in &order[1..] {
            // Search for `ef_construction` nearest among the already-inserted.
            let cands = search_layer(&vectors, &neighbors, &[entry], &vectors[idx as usize], ef_construction);
            let mut cand_sorted: Vec<(f32, u32)> = cands.into_iter().map(|c| (c.dist, c.id)).collect();
            cand_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            // Diverse (heuristic) neighbor selection — canonical HNSW.
            let picked = select_neighbors_heuristic(&vectors, &cand_sorted, m);
            for &nb in &picked {
                neighbors[idx as usize].push(nb);
                neighbors[nb as usize].push(idx);
            }
            for &nb in &picked {
                if neighbors[nb as usize].len() > 2 * m {
                    prune_neighbors_heuristic(&vectors, &mut neighbors, nb, m);
                }
            }
        }

        // Pick 8 highest-degree nodes as entry points — hubs are more
        // effective launch pads for cross-cluster traversal.
        let mut deg: Vec<(usize, u32)> =
            neighbors.iter().enumerate().map(|(i, nbs)| (nbs.len(), i as u32)).collect();
        deg.sort_by(|a, b| b.0.cmp(&a.0));
        let entries: Vec<u32> = deg.into_iter().take(32).map(|(_, i)| i).collect();

        Hnsw { vectors, neighbors, m, entries }
    }

    pub fn len(&self) -> usize { self.vectors.len() }
    pub fn dim(&self) -> usize { self.vectors[0].len() }
    pub fn m(&self) -> usize { self.m }

    /// Best-first search for `k` nearest with the given `ef`. Returns
    /// `(ids, probe_stats)`. `ids` sorted ascending by distance.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> (Vec<u32>, ProbeStats) {
        let ef = ef.max(k);
        let mut stats = ProbeStats::default();

        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: BinaryHeap<MinItem> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxItem> = BinaryHeap::new();

        // Seed with all entry points; d_entry = distance to the *best* entry.
        let mut best_entry_d = f32::INFINITY;
        for &e in &self.entries {
            if visited[e as usize] { continue; }
            visited[e as usize] = true;
            let d = l2_sq(query, &self.vectors[e as usize]);
            stats.dists += 1;
            if d < best_entry_d { best_entry_d = d; }
            candidates.push(MinItem { dist: d, id: e });
            results.push(MaxItem { dist: d, id: e });
            if results.len() > ef { results.pop(); }
        }
        stats.d_entry = best_entry_d;

        // Track first-hop signal from entry's neighbors.
        let mut first_hop_mean = 0.0f32;
        let mut first_hop_count = 0u32;
        let mut first_expansion = true;

        while let Some(top) = candidates.pop() {
            // Termination: current frontier candidate farther than the ef-th result.
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if top.dist > worst && results.len() >= ef {
                break;
            }

            for &nb in &self.neighbors[top.id as usize] {
                if visited[nb as usize] { continue; }
                visited[nb as usize] = true;
                let d = l2_sq(query, &self.vectors[nb as usize]);
                stats.dists += 1;
                if first_expansion {
                    first_hop_mean += d;
                    first_hop_count += 1;
                }
                let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
                if results.len() < ef || d < worst {
                    candidates.push(MinItem { dist: d, id: nb });
                    results.push(MaxItem { dist: d, id: nb });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
            first_expansion = false;
        }

        if first_hop_count > 0 {
            stats.first_hop_mean = first_hop_mean / first_hop_count as f32;
        } else {
            stats.first_hop_mean = best_entry_d;
        }

        // Extract sorted results.
        let mut all: Vec<(f32, u32)> = results.into_iter().map(|r| (r.dist, r.id)).collect();
        all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        if all.len() >= 1 { stats.d1 = all[0].0; }
        if all.len() >= 2 { stats.d2 = all[1].0; } else { stats.d2 = stats.d1; }
        let ids: Vec<u32> = all.into_iter().take(k).map(|(_, i)| i).collect();
        (ids, stats)
    }

    /// Cheap probe (small ef) purely to gather features for a controller.
    /// Returns the ProbeStats; caller can then re-run `search` with a
    /// controller-chosen ef.
    pub fn probe(&self, query: &[f32]) -> ProbeStats {
        let (_ids, s) = self.search(query, 10, 16);
        s
    }

    /// Ground-truth top-k by brute force. Deterministic. Used for calibration
    /// and recall measurement (NOT for hot-path benchmarks).
    pub fn brute_force(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut pairs: Vec<(f32, u32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(query, v), i as u32))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        pairs.into_iter().take(k).map(|(_, i)| i).collect()
    }
}

fn search_layer(
    vectors: &[Vec<f32>],
    neighbors: &[Vec<u32>],
    entries: &[u32],
    query: &[f32],
    ef: usize,
) -> Vec<MaxItem> {
    let mut visited = vec![false; vectors.len()];
    let mut candidates: BinaryHeap<MinItem> = BinaryHeap::new();
    let mut results: BinaryHeap<MaxItem> = BinaryHeap::new();
    for &e in entries {
        if visited[e as usize] { continue; }
        visited[e as usize] = true;
        let d = l2_sq(query, &vectors[e as usize]);
        candidates.push(MinItem { dist: d, id: e });
        results.push(MaxItem { dist: d, id: e });
        if results.len() > ef { results.pop(); }
    }
    while let Some(top) = candidates.pop() {
        let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
        if top.dist > worst && results.len() >= ef { break; }
        for &nb in &neighbors[top.id as usize] {
            if visited[nb as usize] { continue; }
            visited[nb as usize] = true;
            let d = l2_sq(query, &vectors[nb as usize]);
            let worst = results.peek().map(|r| r.dist).unwrap_or(f32::INFINITY);
            if results.len() < ef || d < worst {
                candidates.push(MinItem { dist: d, id: nb });
                results.push(MaxItem { dist: d, id: nb });
                if results.len() > ef { results.pop(); }
            }
        }
    }
    results.into_iter().collect()
}

/// Malkov–Yashunin diverse neighbor selection heuristic. Given candidates
/// sorted ascending by distance-to-node, pick a diverse subset of size `m`:
/// a candidate `c` is accepted iff no already-picked neighbor `r` is
/// strictly closer to `c` than `c` is to the node. Falls back to filling
/// with remaining nearest if the heuristic is too aggressive.
fn select_neighbors_heuristic(
    vectors: &[Vec<f32>],
    cand_sorted: &[(f32, u32)],
    m: usize,
) -> Vec<u32> {
    let mut picked: Vec<u32> = Vec::with_capacity(m);
    let mut picked_dists: Vec<f32> = Vec::with_capacity(m); // d(cand, self) at pick time
    for &(d_c, c) in cand_sorted {
        if picked.len() >= m { break; }
        let mut diverse = true;
        for &r in &picked {
            let d_cr = l2_sq(&vectors[c as usize], &vectors[r as usize]);
            if d_cr < d_c { diverse = false; break; }
        }
        if diverse {
            picked.push(c);
            picked_dists.push(d_c);
        }
    }
    if picked.len() < m {
        for &(_, c) in cand_sorted {
            if picked.len() >= m { break; }
            if !picked.contains(&c) { picked.push(c); }
        }
    }
    picked
}

fn prune_neighbors_heuristic(vectors: &[Vec<f32>], neighbors: &mut Vec<Vec<u32>>, node: u32, m: usize) {
    let nbs = std::mem::take(&mut neighbors[node as usize]);
    let center = &vectors[node as usize];
    let mut with_dist: Vec<(f32, u32)> =
        nbs.into_iter().map(|nb| (l2_sq(center, &vectors[nb as usize]), nb)).collect();
    with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let picked = select_neighbors_heuristic(vectors, &with_dist, m);
    neighbors[node as usize] = picked;
}
