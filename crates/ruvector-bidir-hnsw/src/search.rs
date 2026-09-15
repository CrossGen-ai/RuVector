//! Query-time beam search with three neighbor-expansion modes.

use crate::graph::{l2_sq, BidirGraph};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Vanilla forward-only expansion.
    Baseline,
    /// Expand candidates by Adj[u] ∪ RevAdj[u].
    ExpandReverse,
    /// Graph was pre-symmetrised at build time; forward expansion suffices.
    SymmetrizeBuild,
}

#[derive(Debug, Default, Clone)]
pub struct SearchStats {
    pub distance_evals: usize,
    pub nodes_visited: usize,
}

pub fn search(
    g: &BidirGraph,
    vectors: &[Vec<f32>],
    q: &[f32],
    k: usize,
    ef: usize,
    mode: Mode,
) -> (Vec<(f32, usize)>, SearchStats) {
    let n = g.n;
    let mut stats = SearchStats::default();
    let mut visited = vec![false; n];
    let entry = g.entry;
    let entry_d = l2_sq(q, &vectors[entry]);
    stats.distance_evals += 1;
    visited[entry] = true;
    let mut frontier: Vec<(f32, usize)> = vec![(entry_d, entry)];
    let mut best: Vec<(f32, usize)> = vec![(entry_d, entry)];
    while let Some((d_u, u)) = pop_min(&mut frontier) {
        let worst = best.last().map(|x| x.0).unwrap_or(f32::MAX);
        if d_u > worst && best.len() >= ef {
            break;
        }
        stats.nodes_visited += 1;
        // enumerate neighbor set for this mode.
        let expand: Box<dyn Iterator<Item = usize>> = match mode {
            Mode::Baseline | Mode::SymmetrizeBuild => Box::new(g.neighbors(u)),
            Mode::ExpandReverse => Box::new(g.neighbors(u).chain(g.rev_neighbors(u))),
        };
        for v in expand {
            if visited[v] {
                continue;
            }
            visited[v] = true;
            let d_v = l2_sq(q, &vectors[v]);
            stats.distance_evals += 1;
            let worst_now = best.last().map(|x| x.0).unwrap_or(f32::MAX);
            if best.len() < ef || d_v < worst_now {
                frontier.push((d_v, v));
                insert_sorted(&mut best, (d_v, v), ef);
            }
        }
    }
    best.truncate(k);
    (best, stats)
}

fn pop_min(v: &mut Vec<(f32, usize)>) -> Option<(f32, usize)> {
    if v.is_empty() {
        return None;
    }
    let mut min_i = 0;
    for i in 1..v.len() {
        if v[i].0 < v[min_i].0 {
            min_i = i;
        }
    }
    Some(v.swap_remove(min_i))
}

fn insert_sorted(list: &mut Vec<(f32, usize)>, item: (f32, usize), cap: usize) {
    let pos = list.partition_point(|x| x.0 <= item.0);
    list.insert(pos, item);
    if list.len() > cap {
        list.truncate(cap);
    }
}
