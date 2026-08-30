//! Full multi-layer HNSW query with instrumentation.

use crate::hnsw::HnswIndex;

#[derive(Debug, Default, Clone)]
pub struct SearchStats {
    pub nodes_visited: u64,
    pub distance_calls: u64,
}

/// Top-k search across all layers. Returns `(distance, id)` in ascending
/// distance. Instrumentation is filled in `stats`.
pub fn search(idx: &HnswIndex, q: &[f32], k: usize, ef: usize, stats: &mut SearchStats) -> Vec<(f32, u32)> {
    let Some(ep0) = idx.entry_point else { return Vec::new(); };
    let mut ep = ep0;
    let mut ep_d = crate::sq_l2(idx.get(ep), q);
    stats.distance_calls += 1;

    for l in (1..=idx.top_layer).rev() {
        // Greedy descend
        loop {
            let mut best = ep;
            let mut best_d = ep_d;
            for &n in &idx.layers[l][ep as usize] {
                let d = crate::sq_l2(idx.get(n), q);
                stats.distance_calls += 1;
                stats.nodes_visited += 1;
                if d < best_d {
                    best_d = d;
                    best = n;
                }
            }
            if best == ep {
                break;
            }
            ep = best;
            ep_d = best_d;
        }
    }
    // Layer 0 ef-search.
    use std::cmp::Reverse;
    use crate::hnsw::OrdF32;
    let mut visited = vec![false; idx.len()];
    visited[ep as usize] = true;
    let mut frontier: std::collections::BinaryHeap<Reverse<(OrdF32, u32)>> = std::collections::BinaryHeap::new();
    let mut best: std::collections::BinaryHeap<(OrdF32, u32)> = std::collections::BinaryHeap::new();
    frontier.push(Reverse((OrdF32(ep_d), ep)));
    best.push((OrdF32(ep_d), ep));
    while let Some(Reverse((OrdF32(d), node))) = frontier.pop() {
        let worst = best.peek().map(|x| x.0 .0).unwrap_or(f32::MAX);
        if d > worst && best.len() >= ef {
            break;
        }
        for &nb in &idx.layers[0][node as usize] {
            if visited[nb as usize] {
                continue;
            }
            visited[nb as usize] = true;
            stats.nodes_visited += 1;
            let dd = crate::sq_l2(idx.get(nb), q);
            stats.distance_calls += 1;
            let worst = best.peek().map(|x| x.0 .0).unwrap_or(f32::MAX);
            if best.len() < ef || dd < worst {
                frontier.push(Reverse((OrdF32(dd), nb)));
                best.push((OrdF32(dd), nb));
                if best.len() > ef {
                    best.pop();
                }
            }
        }
    }
    let mut out: Vec<(f32, u32)> = best.into_iter().map(|(d, n)| (d.0, n)).collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    out.truncate(k);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::{HnswConfig, HnswIndex};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn recall_reasonable_on_random_data() {
        let d = 32;
        let n = 500;
        let mut rng = StdRng::seed_from_u64(7);
        let vs: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let mut idx = HnswIndex::new(d, HnswConfig::default());
        for v in &vs { idx.insert(v); }
        let q: Vec<f32> = (0..d).map(|_| rng.gen::<f32>()).collect();
        // brute-force top-10
        let mut brute: Vec<(f32, u32)> = vs.iter()
            .enumerate()
            .map(|(i, v)| (crate::sq_l2(v, &q), i as u32))
            .collect();
        brute.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let truth: std::collections::HashSet<u32> = brute[..10].iter().map(|&(_, i)| i).collect();
        let mut stats = SearchStats::default();
        let ann = search(&idx, &q, 10, 50, &mut stats);
        let hits = ann.iter().filter(|&&(_, id)| truth.contains(&id)).count();
        assert!(hits >= 8, "expected recall >= 8/10, got {}/10", hits);
    }
}
