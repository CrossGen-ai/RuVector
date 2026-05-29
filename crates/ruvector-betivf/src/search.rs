//! Three swappable search strategies on the same IVF backbone.
//!
//! * `FixedNprobe(n)`   — classic IVF: scan top-n closest partitions.
//! * `FixedBudget(b)`   — scan partitions in order until `b` vectors visited.
//! * `BoundedEarlyTerm` — adaptive: stop when the worst current top-k distance
//!   beats the tightest lower bound on any unvisited partition. The bound
//!   uses the triangle inequality:
//!     `min d(q, x in P_c)  >=  max(0, d(q, c) - radius_c)`.
//!   Sound at `slack = 1.0`: never prunes a partition that could improve the result.
//!   `slack > 1.0` widens the stop condition (`lb_sq * slack >= worst_sq`) for
//!   aggressive pruning at some recall cost; `slack < 1.0` is more conservative.

use crate::{l2, l2_sq, IvfIndex};

#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    FixedNprobe(usize),
    FixedBudget(usize),
    BoundedEarlyTerm { max_nprobe: usize, slack: f32 },
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchStats {
    pub partitions_scanned: usize,
    pub vectors_scored: usize,
    pub stopped_early: bool,
}

#[derive(Debug, Clone, Copy)]
struct Hit {
    id: u32,
    dist_sq: f32,
}

pub fn search(
    index: &IvfIndex,
    query: &[f32],
    k: usize,
    strategy: SearchStrategy,
) -> (Vec<(u32, f32)>, SearchStats) {
    assert_eq!(query.len(), index.dim);
    assert!(k > 0);

    let n_parts = index.partitions.len();
    let mut part_order: Vec<(usize, f32, f32)> = Vec::with_capacity(n_parts);
    for (i, p) in index.partitions.iter().enumerate() {
        let dc = l2(query, &p.centroid);
        let lb = (dc - p.radius).max(0.0);
        part_order.push((i, dc, lb * lb));
    }
    part_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut heap: Vec<Hit> = Vec::with_capacity(k + 1);
    let mut worst_sq = f32::INFINITY;
    let mut stats = SearchStats::default();

    let push = |heap: &mut Vec<Hit>, worst_sq: &mut f32, hit: Hit, k: usize| {
        if heap.len() < k {
            heap.push(hit);
            if heap.len() == k {
                *worst_sq = heap.iter().map(|h| h.dist_sq).fold(0.0f32, f32::max);
            }
        } else if hit.dist_sq < *worst_sq {
            let mut worst_idx = 0;
            for i in 1..heap.len() {
                if heap[i].dist_sq > heap[worst_idx].dist_sq {
                    worst_idx = i;
                }
            }
            heap[worst_idx] = hit;
            *worst_sq = heap.iter().map(|h| h.dist_sq).fold(0.0f32, f32::max);
        }
    };

    let (limit, mode) = match strategy {
        SearchStrategy::FixedNprobe(np) => (np.min(n_parts), Mode::FixedN),
        SearchStrategy::FixedBudget(_) => (n_parts, Mode::FixedB),
        SearchStrategy::BoundedEarlyTerm { max_nprobe, .. } => {
            (max_nprobe.min(n_parts), Mode::Bet)
        }
    };
    let budget = match strategy {
        SearchStrategy::FixedBudget(b) => b,
        _ => usize::MAX,
    };
    let slack = match strategy {
        SearchStrategy::BoundedEarlyTerm { slack, .. } => slack,
        _ => 1.0,
    };

    for step in 0..limit {
        let (pi, _dc, lb_sq) = part_order[step];

        if matches!(mode, Mode::Bet) && heap.len() == k && lb_sq * slack >= worst_sq {
            stats.stopped_early = true;
            break;
        }

        let p = &index.partitions[pi];
        stats.partitions_scanned += 1;
        for &m in &p.members {
            let v = index.vector(m);
            let d = l2_sq(query, v);
            stats.vectors_scored += 1;
            push(&mut heap, &mut worst_sq, Hit { id: m, dist_sq: d }, k);
            if matches!(mode, Mode::FixedB) && stats.vectors_scored >= budget {
                break;
            }
        }
        if matches!(mode, Mode::FixedB) && stats.vectors_scored >= budget {
            break;
        }
    }

    heap.sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
    let out: Vec<(u32, f32)> = heap.into_iter().map(|h| (h.id, h.dist_sq)).collect();
    (out, stats)
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    FixedN,
    FixedB,
    Bet,
}

pub fn brute_force_topk(index: &IvfIndex, query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = (0..index.n as u32)
        .map(|i| (i, l2_sq(query, index.vector(i))))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all
}
