//! Bounded max-heap of (dist, id) pairs with a "new/old" flag, the data
//! structure NN-Descent maintains at each node.
//!
//! Why a max-heap?  We want to keep the `k` *closest* neighbours and reject
//! anything worse than the current worst. The worst element sits at the root
//! of a max-heap, giving us O(log k) updates and O(1) "would this improve?"
//! checks.
//!
//! The `new` flag tracks whether a neighbour has participated in a local
//! join yet — this is the trick that drops NN-Descent's distance-call count
//! from O(N·K²) per iteration to "only between recently changed pairs".

use crate::Neighbor;

#[derive(Debug, Clone, Copy)]
pub struct Entry {
    pub id: u32,
    pub dist: f32,
    pub is_new: bool,
}

/// Bounded max-heap by `dist`. Capacity `k` is fixed at construction.
#[derive(Debug, Clone)]
pub struct BoundedMaxHeap {
    cap: usize,
    pub heap: Vec<Entry>,
}

impl BoundedMaxHeap {
    pub fn new(cap: usize) -> Self {
        Self { cap, heap: Vec::with_capacity(cap) }
    }

    pub fn len(&self) -> usize { self.heap.len() }
    pub fn is_empty(&self) -> bool { self.heap.is_empty() }

    /// Worst (largest) distance currently held, or +inf if not full.
    #[inline]
    pub fn worst(&self) -> f32 {
        if self.heap.len() < self.cap {
            f32::INFINITY
        } else {
            self.heap[0].dist
        }
    }

    /// Returns true if `(id, dist)` was inserted (i.e. improved the heap).
    /// Duplicate IDs are filtered.
    pub fn push(&mut self, id: u32, dist: f32, is_new: bool) -> bool {
        // Reject obviously-worse before scanning for duplicates.
        if self.heap.len() == self.cap && dist >= self.heap[0].dist {
            return false;
        }
        for e in &self.heap {
            if e.id == id { return false; }
        }
        if self.heap.len() < self.cap {
            self.heap.push(Entry { id, dist, is_new });
            let last = self.heap.len() - 1;
            sift_up(&mut self.heap, last);
        } else {
            self.heap[0] = Entry { id, dist, is_new };
            sift_down(&mut self.heap, 0);
        }
        true
    }

    /// Drain into a distance-sorted `Vec<Neighbor>` (ascending).
    pub fn into_sorted(mut self) -> Vec<Neighbor> {
        // Heap-sort by repeatedly popping the max.
        let mut out = Vec::with_capacity(self.heap.len());
        while !self.heap.is_empty() {
            let last = self.heap.len() - 1;
            self.heap.swap(0, last);
            let top = self.heap.pop().unwrap();
            out.push(Neighbor { id: top.id, dist: top.dist });
            sift_down(&mut self.heap, 0);
        }
        out.reverse();
        out
    }

    /// Partition the heap's IDs into (new-flagged, old-flagged) and clear
    /// the new flags. Called once per iteration before the local join.
    pub fn split_new_old(&mut self, max_new: usize) -> (Vec<u32>, Vec<u32>) {
        let mut new_ids = Vec::new();
        let mut old_ids = Vec::new();
        for e in &mut self.heap {
            if e.is_new {
                if new_ids.len() < max_new {
                    new_ids.push(e.id);
                    e.is_new = false;
                } else {
                    old_ids.push(e.id);
                }
            } else {
                old_ids.push(e.id);
            }
        }
        (new_ids, old_ids)
    }
}

fn sift_up(h: &mut [Entry], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if h[i].dist > h[parent].dist {
            h.swap(i, parent);
            i = parent;
        } else { break; }
    }
}

fn sift_down(h: &mut [Entry], mut i: usize) {
    let n = h.len();
    loop {
        let l = 2*i + 1;
        let r = 2*i + 2;
        let mut largest = i;
        if l < n && h[l].dist > h[largest].dist { largest = l; }
        if r < n && h[r].dist > h[largest].dist { largest = r; }
        if largest == i { break; }
        h.swap(i, largest);
        i = largest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_keeps_k_smallest() {
        let mut h = BoundedMaxHeap::new(3);
        for (id, d) in [(0,5.0), (1,2.0), (2,8.0), (3,1.0), (4,4.0)] {
            h.push(id, d, true);
        }
        let sorted = h.into_sorted();
        let dists: Vec<f32> = sorted.iter().map(|n| n.dist).collect();
        assert_eq!(dists, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn heap_rejects_duplicates() {
        let mut h = BoundedMaxHeap::new(3);
        assert!(h.push(7, 1.0, true));
        assert!(!h.push(7, 0.5, true));
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn worst_is_inf_when_not_full() {
        let mut h = BoundedMaxHeap::new(3);
        assert!(h.worst().is_infinite());
        h.push(0, 1.0, true);
        assert!(h.worst().is_infinite());
        h.push(1, 2.0, true);
        h.push(2, 3.0, true);
        assert_eq!(h.worst(), 3.0);
    }
}
