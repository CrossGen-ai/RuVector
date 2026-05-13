//! Approximate k-NN graph construction via NN-Descent.
//!
//! Reference: Dong, Charikar & Li, *"Efficient k-nearest neighbor graph
//! construction for generic similarity measures"*, WWW 2011.

use crate::l2_sq;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// One directed edge `(neighbor_id, distance, is_new)`.
#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub id: u32,
    pub dist: f32,
    /// "new" flag for NN-Descent local-join — set on insert, cleared after a
    /// node is used as the source of a join.
    pub fresh: bool,
}

/// Sorted-by-distance bounded neighbor list with `O(K)` insert.
#[derive(Clone, Debug)]
pub struct Heap {
    pub k: usize,
    pub items: Vec<Edge>,
}

impl Heap {
    pub fn new(k: usize) -> Self {
        Self { k, items: Vec::with_capacity(k + 1) }
    }

    /// Insert `(id, dist)` if it improves the list. Returns `true` if inserted.
    pub fn insert(&mut self, id: u32, dist: f32) -> bool {
        // Reject duplicates and over-distance candidates.
        if self.items.len() >= self.k {
            if dist >= self.items.last().unwrap().dist {
                return false;
            }
        }
        // Linear scan — K is tiny (≤ 50).
        let mut pos = self.items.len();
        for i in 0..self.items.len() {
            if self.items[i].id == id {
                return false;
            }
            if pos == self.items.len() && dist < self.items[i].dist {
                pos = i;
            }
        }
        self.items.insert(pos, Edge { id, dist, fresh: true });
        if self.items.len() > self.k {
            self.items.pop();
        }
        true
    }

    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.items.iter().map(|e| e.id)
    }
}

/// Build an approximate K-nearest-neighbor graph via NN-Descent.
///
/// - `k`: out-degree of the kNN graph.
/// - `iters`: NN-Descent sweep count. 4-6 is plenty in practice.
/// - `sample_rate`: fraction of "new" neighbors used per local join (0.4-1.0).
///   Lower = faster, less recall in the kNN graph itself.
pub fn nn_descent(
    data: &[Vec<f32>],
    k: usize,
    iters: usize,
    sample_rate: f32,
    seed: u64,
) -> Vec<Heap> {
    let n = data.len();
    assert!(k < n, "k must be < n");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut graph: Vec<Heap> = (0..n).map(|_| Heap::new(k)).collect();

    // Random initial graph.
    let ids: Vec<u32> = (0..n as u32).collect();
    for i in 0..n {
        let mut sample = ids.clone();
        sample.shuffle(&mut rng);
        for &j in sample.iter().filter(|&&j| j as usize != i).take(k) {
            let d = l2_sq(&data[i], &data[j as usize]);
            graph[i].insert(j, d);
        }
    }

    for _ in 0..iters {
        // Build forward + reverse neighbor lists (B ∪ R per the NN-Descent
        // paper, §3.2). Reverse neighbors are essential — without them
        // pairs that "should" be close never get proposed to each other on
        // datasets where the random init is far from the true kNN.
        let mut new_fwd: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut old_fwd: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n {
            for e in &graph[i].items {
                if e.fresh {
                    new_fwd[i].push(e.id);
                } else {
                    old_fwd[i].push(e.id);
                }
            }
            for e in &mut graph[i].items {
                e.fresh = false;
            }
        }
        let mut new_rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut old_rev: Vec<Vec<u32>> = vec![Vec::new(); n];
        for i in 0..n {
            for &j in &new_fwd[i] {
                new_rev[j as usize].push(i as u32);
            }
            for &j in &old_fwd[i] {
                old_rev[j as usize].push(i as u32);
            }
        }

        let mut proposals: Vec<(u32, u32)> = Vec::with_capacity(n * k);
        for i in 0..n {
            // Combine forward + reverse, sub-sample for speed.
            let cap_new = ((k as f32) * sample_rate).ceil() as usize;
            let cap_new = cap_new.max(1);
            let mut new_n: Vec<u32> = new_fwd[i].clone();
            new_n.extend_from_slice(&new_rev[i]);
            new_n.shuffle(&mut rng);
            new_n.truncate(cap_new);
            let mut old_n: Vec<u32> = old_fwd[i].clone();
            old_n.extend_from_slice(&old_rev[i]);
            old_n.shuffle(&mut rng);
            old_n.truncate(cap_new);

            for a in 0..new_n.len() {
                for b in (a + 1)..new_n.len() {
                    proposals.push((new_n[a], new_n[b]));
                }
                for &o in &old_n {
                    proposals.push((new_n[a], o));
                }
            }
        }

        if proposals.is_empty() {
            break;
        }

        let mut updates = 0usize;
        for (u, v) in proposals {
            if u == v {
                continue;
            }
            let d = l2_sq(&data[u as usize], &data[v as usize]);
            if graph[u as usize].insert(v, d) {
                updates += 1;
            }
            if graph[v as usize].insert(u, d) {
                updates += 1;
            }
        }
        if updates < (n * k) / 100 {
            // < 1% changes — converged.
            break;
        }
    }

    graph
}

/// Compute dataset centroid as a stand-in for a "navigating" reference point.
pub fn centroid(data: &[Vec<f32>]) -> Vec<f32> {
    assert!(!data.is_empty());
    let d = data[0].len();
    let mut c = vec![0.0f32; d];
    for v in data {
        for i in 0..d {
            c[i] += v[i];
        }
    }
    let n = data.len() as f32;
    for x in &mut c {
        *x /= n;
    }
    c
}

/// Find the base point closest to `target` by greedy walk on the kNN graph,
/// starting from a random seed. Cheap O(degree × steps) approximation —
/// adequate because the navigating node only needs to be *near* the centroid.
pub fn nearest_to(
    data: &[Vec<f32>],
    graph: &[Heap],
    target: &[f32],
    seed: u64,
) -> u32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cur = rng.gen_range(0..data.len() as u32);
    let mut cur_d = l2_sq(&data[cur as usize], target);
    loop {
        let mut next = cur;
        let mut next_d = cur_d;
        for nb in graph[cur as usize].ids() {
            let d = l2_sq(&data[nb as usize], target);
            if d < next_d {
                next = nb;
                next_d = d;
            }
        }
        if next == cur {
            return cur;
        }
        cur = next;
        cur_d = next_d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_keeps_topk() {
        let mut h = Heap::new(3);
        for (id, d) in [(0u32, 5.0f32), (1, 1.0), (2, 4.0), (3, 2.0), (4, 3.0)] {
            h.insert(id, d);
        }
        assert_eq!(h.items.len(), 3);
        assert_eq!(h.items[0].id, 1);
        assert_eq!(h.items[1].id, 3);
        assert_eq!(h.items[2].id, 4);
        // Already-bad candidate rejected.
        assert!(!h.insert(7, 10.0));
        // Duplicate rejected.
        assert!(!h.insert(1, 0.0));
    }

    #[test]
    fn nn_descent_recovers_neighbors_on_grid() {
        // 64 points on a 1-D line — exact 2-NN is trivially known: (i-1, i+1).
        let data: Vec<Vec<f32>> =
            (0..64).map(|i| vec![i as f32]).collect();
        let g = nn_descent(&data, 4, 6, 1.0, 42);
        let mut perfect = 0;
        for i in 1..63 {
            let ids: Vec<u32> = g[i].items.iter().map(|e| e.id).collect();
            if ids.contains(&((i - 1) as u32)) && ids.contains(&((i + 1) as u32)) {
                perfect += 1;
            }
        }
        // Tolerate a couple of misses — NN-Descent is approximate.
        assert!(perfect >= 60, "perfect-2NN count = {perfect}/62");
    }
}
