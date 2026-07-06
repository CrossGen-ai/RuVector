//! Baseline exact visited-set using `std::collections::HashSet<u32>`.
//!
//! This is the reference implementation used by most Rust HNSW crates
//! (`hnsw_rs`, `instant-distance`, custom in-house forks).  It is exact —
//! zero false positives, zero false negatives — but pays for a large per-
//! insert working set (bucket + entry + hash) and heap allocation on grow.

use crate::VisitedSet;
use std::collections::HashSet;

/// Exact visited-set backed by `HashSet<u32>`.
pub struct HashSetVisited {
    inner: HashSet<u32>,
}

impl HashSetVisited {
    /// Construct with a hint of expected inserts (candidate frontier size).
    pub fn with_capacity(cap: usize) -> Self {
        Self { inner: HashSet::with_capacity(cap) }
    }
}

impl Default for HashSetVisited {
    fn default() -> Self {
        Self::with_capacity(256)
    }
}

impl VisitedSet for HashSetVisited {
    #[inline]
    fn mark(&mut self, id: u32) -> bool {
        self.inner.insert(id)
    }

    #[inline]
    fn contains(&self, id: u32) -> bool {
        self.inner.contains(&id)
    }

    fn reset(&mut self) {
        self.inner.clear();
    }

    fn bytes(&self) -> usize {
        // HashMap in std uses hashbrown: ~14 bytes/entry amortized (u32 key + control),
        // plus 48 bytes of struct overhead.  Capacity is the honest signal.
        self.inner.capacity() * 14 + 48
    }

    fn name(&self) -> &'static str {
        "HashSet<u32>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_dedup_reset() {
        let mut v = HashSetVisited::with_capacity(16);
        assert!(v.mark(42));
        assert!(!v.mark(42));
        assert!(v.contains(42));
        v.reset();
        assert!(!v.contains(42));
        assert!(v.mark(42));
    }
}
