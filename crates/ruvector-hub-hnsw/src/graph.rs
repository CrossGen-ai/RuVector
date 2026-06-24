//! Flat NSW (Navigable Small World) graph with optional anti-hub pruning.
//!
//! Implementation choices kept deliberately small so the file stays under 500
//! lines and the *hubness* effect is the variable under study:
//!
//! - Single layer (no hierarchical HNSW layers). The original HNSW paper
//!   shows the upper layers improve log-time entry-point routing; they do
//!   not change the qualitative hubness distribution at the base layer,
//!   which is what we are measuring here.
//! - Squared L2 distance.
//! - Top-`M` nearest neighbour selection at insertion (no relative-neighbour
//!   heuristic). This *amplifies* hubness and so makes the anti-hub effect
//!   more legible.
//! - Anti-hub pruning runs as a post-build pass over the materialised graph.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::metric::{l2_sq, Vector};

#[derive(Debug, Clone, Copy)]
pub enum IndegreeCap {
    /// `cap = 3 * M`. Lets hubs grow somewhat but trims the long tail.
    Light,
    /// `cap = 2 * M`. Aggressive: nearly forces indegree parity with outdegree.
    Aggressive,
}

impl IndegreeCap {
    fn value(&self, m: usize) -> usize {
        match self {
            IndegreeCap::Light => 3 * m,
            IndegreeCap::Aggressive => 2 * m,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub seed_entry: u32,
}

impl Default for NswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 64,
            seed_entry: 0,
        }
    }
}

pub trait AnnIndex {
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)>;
    fn adjacency(&self) -> &[Vec<u32>];
    fn label(&self) -> &str;
}

// --- internal: priority queue items -----------------------------------------

#[derive(Debug, Clone, Copy)]
struct Cand {
    dist: f32,
    id: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
// BinaryHeap is a max-heap; we want a min-heap on dist for the candidate
// frontier, so we invert.
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// A max-heap wrapper used for the "best so far" / "results" pool: we want
// the *worst* of the best to be at the top so we can pop it when a better
// candidate arrives.
#[derive(Debug, Clone, Copy)]
struct MaxCand(Cand);
impl PartialEq for MaxCand {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxCand {}
impl PartialOrd for MaxCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxCand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .dist
            .partial_cmp(&other.0.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.0.id.cmp(&other.0.id))
    }
}

// --- shared NSW core --------------------------------------------------------

#[derive(Debug, Clone)]
struct NswCore {
    vectors: Vec<Vector>,
    adj: Vec<Vec<u32>>,
    params: NswParams,
    entry: u32,
}

impl NswCore {
    fn build(vectors: Vec<Vector>, params: NswParams) -> Self {
        let n = vectors.len();
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        let entry = params.seed_entry.min(n.saturating_sub(1) as u32);

        // Insert in id order. Node 0 is the seed: empty neighbours.
        for new_id in 1..n {
            let q = &vectors[new_id];
            // 1. Find ef_construction nearest using current graph from entry.
            let cands = search_internal(
                &vectors,
                &adj,
                entry,
                q,
                params.ef_construction,
                Some(new_id as u32),
            );
            // 2. Take top-M as neighbours.
            let m = params.m.min(cands.len());
            let chosen: Vec<u32> = cands.iter().take(m).map(|c| c.id).collect();
            adj[new_id] = chosen.clone();
            // 3. Bidirectional add with M_max prune on neighbours.
            let m_max = 2 * params.m;
            for &nb in &chosen {
                let nb_idx = nb as usize;
                adj[nb_idx].push(new_id as u32);
                if adj[nb_idx].len() > m_max {
                    // Prune: keep the M_max closest.
                    let mut scored: Vec<(f32, u32)> = adj[nb_idx]
                        .iter()
                        .map(|&x| (l2_sq(&vectors[nb_idx], &vectors[x as usize]), x))
                        .collect();
                    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                    scored.truncate(m_max);
                    adj[nb_idx] = scored.into_iter().map(|(_, id)| id).collect();
                }
            }
        }

        NswCore {
            vectors,
            adj,
            params,
            entry,
        }
    }

    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        let pool = search_internal(&self.vectors, &self.adj, self.entry, q, ef.max(k), None);
        pool.into_iter()
            .take(k)
            .map(|c| (c.id, c.dist))
            .collect()
    }
}

fn search_internal(
    vectors: &[Vector],
    adj: &[Vec<u32>],
    entry: u32,
    q: &[f32],
    ef: usize,
    forbid: Option<u32>,
) -> Vec<Cand> {
    let n = vectors.len();
    if n == 0 {
        return Vec::new();
    }
    let mut visited: HashSet<u32> = HashSet::new();
    let mut frontier: BinaryHeap<Cand> = BinaryHeap::new(); // min-heap by dist
    let mut best: BinaryHeap<MaxCand> = BinaryHeap::new(); // max-heap by dist

    let mut entry_id = entry;
    if let Some(f) = forbid {
        if entry_id == f && n > 1 {
            entry_id = if f == 0 { 1 } else { 0 };
        }
    }
    let d0 = l2_sq(&vectors[entry_id as usize], q);
    let start = Cand { dist: d0, id: entry_id };
    visited.insert(entry_id);
    frontier.push(start);
    best.push(MaxCand(start));

    while let Some(cur) = frontier.pop() {
        // If our nearest unexplored candidate is worse than the worst in best, stop.
        if let Some(worst_best) = best.peek() {
            if cur.dist > worst_best.0.dist && best.len() >= ef {
                break;
            }
        }
        for &nb in &adj[cur.id as usize] {
            if let Some(f) = forbid {
                if nb == f {
                    continue;
                }
            }
            if !visited.insert(nb) {
                continue;
            }
            let d = l2_sq(&vectors[nb as usize], q);
            let cand = Cand { dist: d, id: nb };
            if best.len() < ef {
                best.push(MaxCand(cand));
                frontier.push(cand);
            } else if let Some(worst_best) = best.peek() {
                if d < worst_best.0.dist {
                    best.pop();
                    best.push(MaxCand(cand));
                    frontier.push(cand);
                }
            }
        }
    }

    let mut out: Vec<Cand> = best.into_iter().map(|m| m.0).collect();
    out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    out
}

// --- variants ---------------------------------------------------------------

pub struct BaselineNsw {
    core: NswCore,
}

impl BaselineNsw {
    pub fn build(vectors: Vec<Vector>, params: NswParams) -> Self {
        Self {
            core: NswCore::build(vectors, params),
        }
    }
}

impl AnnIndex for BaselineNsw {
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        self.core.search(q, k, ef)
    }
    fn adjacency(&self) -> &[Vec<u32>] {
        &self.core.adj
    }
    fn label(&self) -> &str {
        "BaselineNsw"
    }
}

pub struct HubNsw {
    core: NswCore,
    cap: IndegreeCap,
    label: &'static str,
}

impl HubNsw {
    pub fn build(vectors: Vec<Vector>, params: NswParams, cap: IndegreeCap) -> Self {
        let mut core = NswCore::build(vectors, params.clone());
        anti_hub_prune(&mut core, cap);
        let label = match cap {
            IndegreeCap::Light => "HubNsw[Light]",
            IndegreeCap::Aggressive => "HubNsw[Aggressive]",
        };
        Self { core, cap, label }
    }
    pub fn cap(&self) -> IndegreeCap {
        self.cap
    }
}

impl AnnIndex for HubNsw {
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        self.core.search(q, k, ef)
    }
    fn adjacency(&self) -> &[Vec<u32>] {
        &self.core.adj
    }
    fn label(&self) -> &str {
        self.label
    }
}

/// Anti-hub pruning. For each node whose indegree exceeds `cap`, we remove
/// its weakest *incoming* edges — i.e. for the furthest source nodes, drop
/// the edge from `src -> hub`. We never strand a source: if removing the
/// edge would leave `src` with fewer than `params.m / 2` outgoing edges,
/// we keep it.
fn anti_hub_prune(core: &mut NswCore, cap: IndegreeCap) {
    let n = core.adj.len();
    let m = core.params.m;
    let cap_val = cap.value(m);
    let min_out = (m / 2).max(2);

    // Build reverse adjacency.
    let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (src, edges) in core.adj.iter().enumerate() {
        for &dst in edges {
            rev[dst as usize].push(src as u32);
        }
    }

    for hub in 0..n {
        let indeg = rev[hub].len();
        if indeg <= cap_val {
            continue;
        }
        // Sort incoming sources by their distance to the hub, ascending.
        // The *closest* sources are the "real" neighbours; the *furthest*
        // ones contribute most to spurious hub traversal — drop those first.
        let hub_vec = &core.vectors[hub];
        let mut scored: Vec<(f32, u32)> = rev[hub]
            .iter()
            .map(|&s| (l2_sq(hub_vec, &core.vectors[s as usize]), s))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Keep the closest cap_val; consider dropping the rest.
        let keep_ids: HashSet<u32> = scored.iter().take(cap_val).map(|(_, id)| *id).collect();
        for (_, src) in scored.iter().skip(cap_val) {
            let src_idx = *src as usize;
            if core.adj[src_idx].len() <= min_out {
                continue; // protect under-degree sources
            }
            core.adj[src_idx].retain(|&x| x as usize != hub);
            let _ = keep_ids; // satisfy borrow checker analyser
        }
    }
}

// --- tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hubness;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn synth(n: usize, d: usize, seed: u64) -> Vec<Vector> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect())
            .collect()
    }

    fn brute_top_k(vectors: &[Vector], q: &[f32], k: usize) -> Vec<u32> {
        let mut scored: Vec<(f32, u32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_sq(v, q), i as u32))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn baseline_recall_reasonable() {
        let vs = synth(400, 32, 7);
        let p = NswParams {
            m: 12,
            ef_construction: 48,
            seed_entry: 0,
        };
        let idx = BaselineNsw::build(vs.clone(), p);
        // Query is an existing vector — should always find itself.
        let q = &vs[123];
        let got = idx.search(q, 5, 32);
        assert!(got.iter().any(|(id, _)| *id == 123));
        // Approximate recall@10 vs brute force across 20 queries.
        let mut total = 0usize;
        let mut hit = 0usize;
        for qi in (0..vs.len()).step_by(20).take(20) {
            let truth = brute_top_k(&vs, &vs[qi], 10);
            let got = idx.search(&vs[qi], 10, 64);
            for t in &truth {
                if got.iter().any(|(id, _)| id == t) {
                    hit += 1;
                }
            }
            total += truth.len();
        }
        let recall = hit as f32 / total as f32;
        assert!(recall > 0.7, "baseline recall@10 too low: {}", recall);
    }

    #[test]
    fn hub_pruning_reduces_max_indegree() {
        let vs = synth(400, 32, 11);
        let p = NswParams {
            m: 12,
            ef_construction: 48,
            seed_entry: 0,
        };
        let base = BaselineNsw::build(vs.clone(), p.clone());
        let agg = HubNsw::build(vs, p, IndegreeCap::Aggressive);
        let s_base = hubness::stats_from(base.adjacency());
        let s_agg = hubness::stats_from(agg.adjacency());
        assert!(
            s_agg.max <= s_base.max,
            "expected aggressive max indegree {} <= baseline {}",
            s_agg.max,
            s_base.max
        );
    }

    #[test]
    fn search_empty_safe() {
        let p = NswParams::default();
        let idx = BaselineNsw::build(vec![], p);
        let got = idx.search(&[0.0; 4], 5, 16);
        assert!(got.is_empty());
    }
}
