//! IVF scan over a compressed posting list.
//!
//! Decode the list, then for each id compute squared-L2 to a query against
//! a flat `[f32; dim]` vector table and track the top-K via a simple
//! max-heap of size K.

use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub dist: f32,
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        // NaN-safe: treat NaN as larger (push out of top-K).
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Greater)
    }
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Scan `ids` against `vectors` (flat `n*dim` layout) with query `q`,
/// returning the top-K smallest-distance hits.
pub fn scan(ids: &[u32], vectors: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<Hit> {
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Hit> = BinaryHeap::with_capacity(k + 1);
    for &id in ids {
        let base = (id as usize) * dim;
        let v = &vectors[base..base + dim];
        let d = l2_sq(v, q);
        if heap.len() < k {
            heap.push(Hit { id, dist: d });
        } else if let Some(top) = heap.peek() {
            if d < top.dist {
                heap.pop();
                heap.push(Hit { id, dist: d });
            }
        }
    }
    let mut out: Vec<Hit> = heap.into_vec();
    out.sort_by(|a, b| a.cmp(b));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_basic_topk() {
        let dim = 4;
        let vectors: Vec<f32> = (0..16 * dim)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let q = vec![0.0f32; dim];
        let ids: Vec<u32> = (0..16).collect();
        let hits = scan(&ids, &vectors, dim, &q, 3);
        assert_eq!(hits.len(), 3);
        // Closest to the origin is id=0 (smallest magnitudes).
        assert_eq!(hits[0].id, 0);
    }
}
