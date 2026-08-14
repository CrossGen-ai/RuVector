//! ruvector-visited-filter
//!
//! Pluggable "visited set" implementations for graph ANN traversal (HNSW,
//! NSG, DiskANN and friends). During a single search the algorithm walks
//! thousands of nodes and asks "have I already looked at node `u`?" on every
//! hop. The data structure behind that question dominates the inner loop
//! once distance evaluations are cheap (quantized codes, cached vectors).
//!
//! This crate ships three interchangeable backends behind a common
//! `VisitedFilter` trait, plus a `SearchScratch` pool that lets callers
//! reuse allocations across queries — exactly how a production ANN engine
//! would wire it in.
//!
//! Backends
//! --------
//! * [`HashSetVisited`]  — the naive baseline (`hashbrown`-style hashmap
//!   would be equivalent). Zeroed cost on `new_search`, O(1) amortized ops,
//!   but poor cache behavior on skewed graphs.
//! * [`BitmapVisited`]   — a `Vec<u64>` dense bitmap. Constant-time ops,
//!   perfect cache locality, but must `memset` the entire mask each search.
//! * [`GenerationVisited`] — a `Vec<u32>` of per-node "last visited"
//!   generation tags. `new_search` bumps a counter (O(1)); a node is
//!   "visited" iff its tag equals the current generation. Wraps safely
//!   using a full reset on overflow.
//!
//! The trait keeps zero unsafe code, no external allocations per query
//! (after warmup) and works for graphs up to `u32::MAX` nodes.

#![forbid(unsafe_code)]

use std::collections::HashSet;

/// Interface every visited-set backend implements.
pub trait VisitedFilter {
    /// Begin a fresh search. MUST be O(1) or amortized O(1); anything
    /// worse defeats the point of pooling the scratch.
    fn new_search(&mut self);
    /// Mark `u` as visited. Returns `true` iff `u` was already visited.
    fn insert(&mut self, u: u32) -> bool;
    /// Query without mutating.
    fn contains(&self, u: u32) -> bool;
    /// Backend identifier for reports.
    fn name(&self) -> &'static str;
    /// Estimated bytes of internal storage (excluding self).
    fn bytes(&self) -> usize;
}

// ---------------------------------------------------------------- HashSet

/// Baseline: `std::collections::HashSet<u32>`. Clears per search.
#[derive(Default)]
pub struct HashSetVisited {
    inner: HashSet<u32>,
}
impl HashSetVisited {
    pub fn new() -> Self { Self::default() }
}
impl VisitedFilter for HashSetVisited {
    #[inline] fn new_search(&mut self) { self.inner.clear(); }
    #[inline] fn insert(&mut self, u: u32) -> bool { !self.inner.insert(u) }
    #[inline] fn contains(&self, u: u32) -> bool { self.inner.contains(&u) }
    fn name(&self) -> &'static str { "hashset" }
    fn bytes(&self) -> usize {
        // rough: HashSet capacity * (u32 + control byte)
        self.inner.capacity() * (std::mem::size_of::<u32>() + 1)
    }
}

// ---------------------------------------------------------------- Bitmap

/// Dense `u64` bitmap over the node space. `new_search` clears the mask.
pub struct BitmapVisited {
    bits: Vec<u64>,
    capacity: u32,
}
impl BitmapVisited {
    pub fn new(capacity: u32) -> Self {
        let words = ((capacity as usize) + 63) / 64;
        Self { bits: vec![0u64; words], capacity }
    }
    #[inline] fn split(u: u32) -> (usize, u64) {
        ((u >> 6) as usize, 1u64 << (u & 63))
    }
}
impl VisitedFilter for BitmapVisited {
    #[inline]
    fn new_search(&mut self) {
        // Vec<u64>::fill compiles to memset on x86_64/aarch64.
        self.bits.fill(0);
    }
    #[inline]
    fn insert(&mut self, u: u32) -> bool {
        debug_assert!(u < self.capacity);
        let (w, m) = Self::split(u);
        let was = (self.bits[w] & m) != 0;
        self.bits[w] |= m;
        was
    }
    #[inline]
    fn contains(&self, u: u32) -> bool {
        let (w, m) = Self::split(u);
        (self.bits[w] & m) != 0
    }
    fn name(&self) -> &'static str { "bitmap" }
    fn bytes(&self) -> usize { self.bits.len() * 8 }
}

// -------------------------------------------------------------- Generation

/// Generation-tagged filter. Each slot stores the last search generation
/// that touched the node; a slot "matches" the current search iff its tag
/// equals `gen`. `new_search` is O(1) — just bump the counter — except on
/// the rare overflow (~ every 4 billion searches) where we reset the tags.
pub struct GenerationVisited {
    tags: Vec<u32>,
    gen: u32,
}
impl GenerationVisited {
    pub fn new(capacity: u32) -> Self {
        Self { tags: vec![0u32; capacity as usize], gen: 0 }
    }
}
impl VisitedFilter for GenerationVisited {
    #[inline]
    fn new_search(&mut self) {
        // Reserve `0` as "never visited" so the initial state is valid.
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.tags.fill(0);
            self.gen = 1;
        }
    }
    #[inline]
    fn insert(&mut self, u: u32) -> bool {
        let slot = &mut self.tags[u as usize];
        let was = *slot == self.gen;
        *slot = self.gen;
        was
    }
    #[inline]
    fn contains(&self, u: u32) -> bool {
        self.tags[u as usize] == self.gen
    }
    fn name(&self) -> &'static str { "generation" }
    fn bytes(&self) -> usize { self.tags.len() * 4 }
}

// ---------------------------------------------------------------- Scratch

/// A pooled search scratch holding any [`VisitedFilter`]. Real ANN
/// engines keep a `thread_local` scratch to avoid per-query alloc.
pub struct SearchScratch<F: VisitedFilter> {
    pub filter: F,
    pub touched: u64,
}
impl<F: VisitedFilter> SearchScratch<F> {
    pub fn new(filter: F) -> Self { Self { filter, touched: 0 } }
    /// Simulate a single ANN search: walk the given visit stream and
    /// return the number of nodes that were *newly* visited.
    pub fn simulate<I: IntoIterator<Item = u32>>(&mut self, stream: I) -> u64 {
        self.filter.new_search();
        let mut newly = 0u64;
        for u in stream {
            if !self.filter.insert(u) { newly += 1; }
        }
        self.touched = self.touched.wrapping_add(newly);
        newly
    }
}

// ---------------------------------------------------------------- Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn conformance<F: VisitedFilter>(mut f: F) {
        // Fresh search: nothing visited.
        f.new_search();
        assert!(!f.contains(0));
        assert!(!f.insert(7));    // fresh -> not previously visited
        assert!(f.contains(7));
        assert!(f.insert(7));     // duplicate insert -> was visited
        assert!(!f.insert(9));
        // A new search must forget prior nodes.
        f.new_search();
        assert!(!f.contains(7));
        assert!(!f.contains(9));
    }

    #[test]
    fn hashset_conforms() { conformance(HashSetVisited::new()); }

    #[test]
    fn bitmap_conforms() { conformance(BitmapVisited::new(64)); }

    #[test]
    fn generation_conforms() { conformance(GenerationVisited::new(64)); }

    #[test]
    fn generation_survives_overflow() {
        // Force the wraparound branch and verify correctness after it.
        let mut f = GenerationVisited::new(32);
        f.gen = u32::MAX - 1;
        f.new_search();          // -> u32::MAX
        assert!(!f.insert(3));
        assert!(f.contains(3));
        f.new_search();          // -> 0, triggers reset -> 1
        assert_eq!(f.gen, 1);
        assert!(!f.contains(3));
    }

    #[test]
    fn scratch_counts_new_visits() {
        let mut s = SearchScratch::new(GenerationVisited::new(128));
        let stream = [1u32, 2, 3, 2, 1, 4, 4, 5];
        assert_eq!(s.simulate(stream), 5); // {1,2,3,4,5}
    }
}
