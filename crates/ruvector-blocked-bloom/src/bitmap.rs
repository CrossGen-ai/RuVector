//! Exact dense bitmap visited-set (`Vec<u64>` bit vector indexed by node id).
//!
//! This is the "you know your id space" implementation.  For an HNSW index of
//! `N` nodes the memory cost is a flat `N/8` bytes; a 10 M-node index costs
//! 1.25 MB.  Because the bit for id `x` lives at fixed offset `x/64`, both
//! [`mark`] and [`contains`] are single-cache-line, allocation-free.
//!
//! Reset is O(dirty_words): we track the set of touched words in a small
//! side-buffer so multi-million-node resets stay sub-millisecond even for
//! short queries.

use crate::VisitedSet;

/// Exact bitmap visited-set with O(dirty) reset.
pub struct BitmapVisited {
    /// One bit per node id.  Length is `ceil(n_ids / 64)` words.
    bits: Vec<u64>,
    /// Word indices modified since last reset.  Bounded by the frontier size.
    dirty: Vec<u32>,
}

impl BitmapVisited {
    /// Allocate for an id space of `n_ids` (max id, exclusive).
    pub fn new(n_ids: usize) -> Self {
        let n_words = n_ids.div_ceil(64);
        Self { bits: vec![0u64; n_words], dirty: Vec::with_capacity(1024) }
    }
}

impl VisitedSet for BitmapVisited {
    #[inline]
    fn mark(&mut self, id: u32) -> bool {
        let word_ix = (id >> 6) as usize;
        let bit = 1u64 << (id & 63);
        // Safety-through-bounds: caller must not exceed n_ids passed to `new`.
        // We keep this a checked index so a bad id is a panic, not UB.
        let w = &mut self.bits[word_ix];
        let was_set = (*w & bit) != 0;
        if !was_set {
            *w |= bit;
            self.dirty.push(word_ix as u32);
        }
        !was_set
    }

    #[inline]
    fn contains(&self, id: u32) -> bool {
        let word_ix = (id >> 6) as usize;
        let bit = 1u64 << (id & 63);
        self.bits.get(word_ix).is_some_and(|w| (*w & bit) != 0)
    }

    fn reset(&mut self) {
        // O(dirty) — critical for scenarios where n_ids >> frontier.
        for &ix in &self.dirty {
            self.bits[ix as usize] = 0;
        }
        self.dirty.clear();
    }

    fn bytes(&self) -> usize {
        self.bits.len() * 8 + self.dirty.capacity() * 4 + 48
    }

    fn name(&self) -> &'static str {
        "Bitmap<u64>"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut v = BitmapVisited::new(10_000);
        for id in [3u32, 100, 9_999] {
            assert!(v.mark(id));
            assert!(!v.mark(id));
            assert!(v.contains(id));
        }
        v.reset();
        for id in [3u32, 100, 9_999] {
            assert!(!v.contains(id));
            assert!(v.mark(id));
        }
    }

    #[test]
    fn dirty_reset_is_local() {
        let mut v = BitmapVisited::new(1_000_000);
        v.mark(42);
        v.mark(999_999);
        v.reset();
        assert!(v.dirty.is_empty());
        assert!(!v.contains(42));
        assert!(!v.contains(999_999));
    }
}
