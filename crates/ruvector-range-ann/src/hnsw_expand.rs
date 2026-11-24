//! HNSW-expand: a flat-graph variant where range search is driven by a
//! frontier of in-ball candidates rather than a top-k heap.
//!
//! Construction is a simple navigable-small-world style: each new point
//! greedily searches for its `M` nearest neighbors already in the graph
//! and connects to them (bidirectional). We don't bother with layering
//! because range search is a single-pass flood from a seed; the
//! hierarchical descent that HNSW uses to amortise entry-point selection
//! is replaced by a tiny set of random seed points.
//!
//! The novelty here is the expansion rule:
//!
//! - A visited candidate `p` is **kept** if `dist(q, p) <= r`.
//! - Its neighbors are **expanded** if `dist(q, p) <= r * (1 + alpha)`.
//!
//! This is a strict superset of top-k ANN search: setting `r = kth-nn`
//! recovers the HNSW ef-search behaviour. Taking `alpha > 0` lets the
//! walker cross small "valleys" that are just outside the ball, which
//! is how we trade recall against work.

use crate::{sqr_l2, RangeHit, RangeIndex};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::collections::BinaryHeap;

pub struct HnswExpand {
    points: Vec<Vec<f32>>,
    adjacency: Vec<Vec<u32>>,
    seeds: Vec<u32>,
    pub alpha: f32,
    /// Hard upper bound on nodes expanded per query. Scales with alpha
    /// so a "wider" instance really does spend more work.
    pub max_expand: usize,
    pub expanded_last: std::sync::atomic::AtomicUsize,
}

impl HnswExpand {
    pub fn build_with(points: &[Vec<f32>], m: usize, ef_build: usize, alpha: f32) -> Self {
        let n = points.len();
        assert!(n > 0);
        let mut rng = ChaCha20Rng::seed_from_u64(0xA11CE);
        let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); n];

        // insertion order: shuffled for better graph quality
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            order.swap(i, j);
        }

        // first point has no edges
        let mut inserted = vec![false; n];
        inserted[order[0]] = true;

        for &idx in &order[1..] {
            let candidates = search_layer(&points, &adjacency, &inserted, &points[idx], ef_build, order[0] as u32);
            // keep at most M nearest
            let m_select = m.min(candidates.len());
            let mut cands = candidates;
            cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            cands.truncate(m_select);
            for (nid, _) in &cands {
                adjacency[idx].push(*nid);
                adjacency[*nid as usize].push(idx as u32);
            }
            // prune each touched node to M
            for (nid, _) in cands {
                prune_to(&mut adjacency, nid as usize, m, &points);
            }
            inserted[idx] = true;
        }

        // Add small-world long-range edges. M-nearest-only graphs on
        // cluster-separated data rarely cross cluster boundaries, which
        // strands greedy descent in the seed's cluster. Four random
        // long edges per node restores navigability (Kleinberg-style).
        let n_long = 4usize;
        for i in 0..n {
            for _ in 0..n_long {
                let j = rng.gen_range(0..n);
                if j != i && !adjacency[i].contains(&(j as u32)) {
                    adjacency[i].push(j as u32);
                    adjacency[j].push(i as u32);
                }
            }
        }

        // seeds: enough to hit each cluster with high probability.
        let n_seeds = (4 * ((n as f32).log2().ceil() as usize)).max(16).min(96);
        let mut seeds = Vec::with_capacity(n_seeds);
        for i in 0..n_seeds {
            seeds.push(((i * 2654435761usize) % n) as u32);
        }

        // Expansion budget scales with alpha so a "wider" instance really
        // spends more work. Base = 8*sqrt(n), stretch factor = (1+alpha)^2
        // so that moving alpha from 0.5 -> 1.5 roughly quadruples the
        // budget — matching the geometric volume growth of the expand ball.
        let base = (8.0 * (n as f32).sqrt()) as usize;
        let max_expand = ((base as f32) * (1.0 + alpha) * (1.0 + alpha)) as usize;
        let max_expand = max_expand.max(512).min(n);
        Self {
            points: points.to_vec(),
            adjacency,
            seeds,
            alpha,
            max_expand,
            expanded_last: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

/// Prune the neighbor list of `node` back to `m` by distance.
fn prune_to(adjacency: &mut Vec<Vec<u32>>, node: usize, m: usize, points: &[Vec<f32>]) {
    if adjacency[node].len() <= m {
        return;
    }
    let p = &points[node];
    let mut with_d: Vec<(u32, f32)> = adjacency[node]
        .iter()
        .map(|nid| (*nid, sqr_l2(p, &points[*nid as usize])))
        .collect();
    with_d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    with_d.truncate(m);
    adjacency[node] = with_d.into_iter().map(|(n, _)| n).collect();
}

/// Standard ef-greedy layer search used during construction (not ball-based).
fn search_layer(
    points: &[Vec<f32>],
    adjacency: &[Vec<u32>],
    inserted: &[bool],
    q: &[f32],
    ef: usize,
    entry: u32,
) -> Vec<(u32, f32)> {
    // max-heap of best candidates so far (we store negated for max->min behavior via Ord)
    let mut visited = vec![false; points.len()];
    let d0 = sqr_l2(q, &points[entry as usize]);
    visited[entry as usize] = true;

    // candidates: min-heap by dist (BinaryHeap is max, so negate)
    let mut candidates: BinaryHeap<Ordered> = BinaryHeap::new();
    candidates.push(Ordered { id: entry, neg_d: -d0 });

    // result: max-heap by dist (prefer to kick farthest)
    let mut result: BinaryHeap<Ordered> = BinaryHeap::new();
    result.push(Ordered { id: entry, neg_d: d0 });

    while let Some(c) = candidates.pop() {
        let d_c = -c.neg_d;
        let worst_in_result = result.peek().map(|r| r.neg_d).unwrap_or(f32::INFINITY);
        if d_c > worst_in_result && result.len() >= ef {
            break;
        }
        for &nbr in &adjacency[c.id as usize] {
            if !inserted[nbr as usize] {
                continue;
            }
            if visited[nbr as usize] {
                continue;
            }
            visited[nbr as usize] = true;
            let d_n = sqr_l2(q, &points[nbr as usize]);
            let worst = result.peek().map(|r| r.neg_d).unwrap_or(f32::INFINITY);
            if result.len() < ef || d_n < worst {
                candidates.push(Ordered { id: nbr, neg_d: -d_n });
                result.push(Ordered { id: nbr, neg_d: d_n });
                if result.len() > ef {
                    result.pop();
                }
            }
        }
    }

    result.into_iter().map(|o| (o.id, o.neg_d)).collect()
}

#[derive(Copy, Clone, Debug)]
struct Ordered {
    id: u32,
    neg_d: f32, // overloaded: sometimes "distance" when used in a max-heap of results
}

impl PartialEq for Ordered {
    fn eq(&self, other: &Self) -> bool {
        self.neg_d == other.neg_d
    }
}
impl Eq for Ordered {}
impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.neg_d.partial_cmp(&other.neg_d).unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl RangeIndex for HnswExpand {
    fn build(points: &[Vec<f32>]) -> Self {
        Self::build_with(points, 16, 32, 0.15)
    }

    fn range_search(&self, query: &[f32], r2: f32) -> Vec<RangeHit> {
        let r = r2.sqrt();
        let expand_r = r * (1.0 + self.alpha);
        let expand_r2 = expand_r * expand_r;

        let n = self.points.len();
        let mut visited = vec![false; n];
        let mut result: Vec<RangeHit> = Vec::new();

        // Best-first frontier: min-heap by distance to q. Expansion is
        // bounded by (a) the expand-radius and (b) an overall budget
        // `max_expand`, whichever comes first. Best-first is essential
        // here: we always expand the node currently closest to q, which
        // means our effort is concentrated in the ball even when we
        // enter from a far seed.
        let mut frontier: BinaryHeap<Ordered> = BinaryHeap::new();

        // Seed each starting point.
        for &s in &self.seeds {
            if !visited[s as usize] {
                visited[s as usize] = true;
                let d = sqr_l2(query, &self.points[s as usize]);
                if d <= r2 {
                    result.push((s, d));
                }
                // Always seed the frontier; the budget bounds work.
                frontier.push(Ordered { id: s, neg_d: -d });
            }
        }

        let mut expanded = 0usize;
        let warmup = self.max_expand / 4;
        while let Some(cur) = frontier.pop() {
            let cur_d = -cur.neg_d;
            if expanded >= self.max_expand {
                break;
            }
            // After warmup, stop once the best remaining frontier node is
            // outside the expand ball — best-first guarantees nothing
            // closer remains.
            if expanded > warmup && cur_d > expand_r2 {
                break;
            }
            expanded += 1;
            for &nbr in &self.adjacency[cur.id as usize] {
                if visited[nbr as usize] {
                    continue;
                }
                visited[nbr as usize] = true;
                let d_n = sqr_l2(query, &self.points[nbr as usize]);
                if d_n <= r2 {
                    result.push((nbr, d_n));
                }
                // During warmup, push every neighbor (long-range navigation).
                // After warmup, push only neighbors that are inside the
                // expand ball (ball-bounded flooding).
                if expanded < warmup || d_n <= expand_r2 {
                    frontier.push(Ordered { id: nbr, neg_d: -d_n });
                }
            }
        }

        self.expanded_last
            .store(expanded, std::sync::atomic::Ordering::Relaxed);
        result
    }

    fn name(&self) -> &'static str {
        "hnsw-expand-ball"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::make_corpus;
    use crate::linear::LinearScan;

    #[test]
    fn hnsw_expand_recovers_most_hits() {
        // Larger corpus + larger std so the cluster blobs overlap enough that
        // a reasonable r2 produces non-empty truth sets.
        let corp = make_corpus(42, 4000, 20, 16, 8, 0.25);
        // In 16 dims with std=0.25 the same-cluster expected sqr distance is
        // 2*dim*std^2 = 2. Pick r2 = 1.0 so we land inside most clusters.
        let r2 = 1.0f32;
        let hnsw = HnswExpand::build_with(&corp.points, 24, 64, 1.5);
        let lin = LinearScan::build(&corp.points);
        let mut total_recall = 0.0f32;
        let mut counted = 0usize;
        for q in &corp.queries {
            let truth = lin.range_search(q, r2);
            if truth.is_empty() {
                continue;
            }
            let got = hnsw.range_search(q, r2);
            let gotset: std::collections::HashSet<u32> = got.iter().map(|(i, _)| *i).collect();
            let hit = truth.iter().filter(|(i, _)| gotset.contains(i)).count();
            total_recall += hit as f32 / truth.len() as f32;
            counted += 1;
        }
        assert!(counted > 0, "truth sets were all empty");
        let mean = total_recall / counted as f32;
        // Sanity bound: approximate range search should retrieve a
        // non-trivial fraction of the ground truth. For higher recall we
        // dial alpha up or use a bigger graph; the bench covers both.
        assert!(mean > 0.5, "expected mean recall > 0.5, got {mean}");
    }

    #[test]
    fn ivf_range_is_exact() {
        use crate::ivf::IvfRange;
        let corp = make_corpus(7, 3000, 15, 12, 6, 0.2);
        // 12 dims * 2 * 0.04 = 0.96 expected intra-cluster sqr distance.
        let r2 = 0.6f32;
        let ivf = IvfRange::build_with(&corp.points, 48, 6);
        let lin = LinearScan::build(&corp.points);
        for q in &corp.queries {
            let mut truth = lin.range_search(q, r2);
            let mut got = ivf.range_search(q, r2);
            truth.sort_by_key(|(i, _)| *i);
            got.sort_by_key(|(i, _)| *i);
            // IVF-with-ball-prune is exact: sets must match as ids
            let t_ids: Vec<u32> = truth.iter().map(|(i, _)| *i).collect();
            let g_ids: Vec<u32> = got.iter().map(|(i, _)| *i).collect();
            assert_eq!(t_ids, g_ids);
        }
    }
}
