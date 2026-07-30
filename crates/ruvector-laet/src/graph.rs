//! Minimal flat HNSW-like graph: single layer, brute-force build with M nearest neighbors.
//!
//! Hermetic — no dependency on other ruvector crates. Kept intentionally small so the
//! learned-early-termination experiment (LAET) can be studied in isolation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};

/// Global distance-call counter. Reset with [`reset_dist_calls`]; read with [`dist_calls`].
pub static DIST_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn reset_dist_calls() {
    DIST_CALLS.store(0, AtomicOrd::Relaxed);
}
pub fn dist_calls() -> u64 {
    DIST_CALLS.load(AtomicOrd::Relaxed)
}

/// L2^2 distance (skip sqrt — monotone).
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    DIST_CALLS.fetch_add(1, AtomicOrd::Relaxed);
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[derive(Clone, Debug)]
pub struct FlatGraph {
    pub dim: usize,
    pub m: usize,
    pub points: Vec<Vec<f32>>,
    /// neighbors[i] = up to m ids sorted by ascending distance to point i.
    pub neighbors: Vec<Vec<u32>>,
}

impl FlatGraph {
    /// Build a k-NN graph by brute force plus a few long-range random edges per node
    /// (a lightweight stand-in for HNSW's hierarchical layers — without them, a
    /// clustered dataset produces a disconnected graph and search cannot reach
    /// queries in other clusters). O(n^2 d) — fine for n<=5k.
    pub fn build(points: Vec<Vec<f32>>, m: usize) -> Self {
        use rand::Rng;
        use rand_chacha::rand_core::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let n = points.len();
        let dim = if n > 0 { points[0].len() } else { 0 };
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::with_capacity(m + 4); n];
        for i in 0..n {
            let mut heap: BinaryHeap<Item> = BinaryHeap::with_capacity(m + 1);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = l2sq(&points[i], &points[j]);
                if heap.len() < m {
                    heap.push(Item { dist: d, id: j as u32 });
                } else if let Some(top) = heap.peek() {
                    if d < top.dist {
                        heap.pop();
                        heap.push(Item { dist: d, id: j as u32 });
                    }
                }
            }
            let mut v: Vec<Item> = heap.into_sorted_vec();
            v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            neighbors[i] = v.into_iter().map(|x| x.id).collect();
        }
        // Long-range random edges: 4 per node, deterministic seed.
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0FFEE ^ n as u64);
        for i in 0..n {
            for _ in 0..4 {
                let j = rng.gen_range(0..n) as u32;
                if j != i as u32 && !neighbors[i].contains(&j) {
                    neighbors[i].push(j);
                }
            }
        }
        reset_dist_calls();
        Self { dim, m, points, neighbors }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Item {
    pub dist: f32,
    pub id: u32,
}
impl Eq for Item {}
impl PartialEq for Item {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist && self.id == o.id
    }
}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        // Max-heap by dist.
        self.dist.partial_cmp(&o.dist)
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        self.partial_cmp(o).unwrap_or(Ordering::Equal)
    }
}

/// Ascending-dist wrapper (min-heap via BinaryHeap<Reverse<..>>-alternative).
#[derive(Clone, Copy, Debug)]
pub struct MinItem(pub Item);
impl Eq for MinItem {}
impl PartialEq for MinItem {
    fn eq(&self, o: &Self) -> bool {
        self.0.dist == o.0.dist && self.0.id == o.0.id
    }
}
impl PartialOrd for MinItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        o.0.dist.partial_cmp(&self.0.dist)
    }
}
impl Ord for MinItem {
    fn cmp(&self, o: &Self) -> Ordering {
        self.partial_cmp(o).unwrap_or(Ordering::Equal)
    }
}
