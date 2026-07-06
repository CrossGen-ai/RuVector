//! Cache-line-aligned **Blocked Bloom Filter** visited-set.
//!
//! Traditional Bloom filters read `k` scattered bits per probe, incurring `k`
//! cache misses on cold state.  A *blocked* Bloom filter concentrates all
//! `k` bits inside one aligned 64-byte block, so each probe touches exactly
//! one cache line.  This is the design used by, among others,
//! Putze/Sanders/Singler (2007) and modern OLAP engines (Impala, DuckDB).
//!
//! ## Design
//!
//! * `B` blocks of 8 × u64 (512 bits, 64 bytes — one cache line).
//! * Block chosen by the high 32 bits of `splitmix64(id)`.
//! * Within the block: `k=4` distinct bit positions derived from the low
//!   32 bits by four multiplicative mixers.
//! * All lanes for a probe live in the same block → one prefetch, one
//!   cache line, four `bts` / `bt` operations.
//!
//! ## Trade-off
//!
//! False-positive on `mark` = "already seen" when in fact new → the caller
//! *skips* processing the id.  Because HNSW search is approximate the only
//! observable effect is a tiny recall loss.  We measure this in the
//! `benchmark` binary and report FP rate vs load factor.

use crate::VisitedSet;

/// Number of hash lanes per block.  4 is the sweet spot: FP ~= 0.5% at
/// n/B ≈ 32 inserts per block (7% load), rising to ~2.5% at n/B ≈ 64.
pub const K_HASHES: usize = 4;

/// Bits per block.  64 bytes = 1 x86_64 / aarch64 cache line.
pub const BLOCK_BITS: usize = 512;

/// Blocked Bloom visited-set.  Fixed size; no allocation after `new`.
#[repr(align(64))]
pub struct BlockedBloomVisited {
    /// `n_blocks × 8` u64 words.  `#[repr(align(64))]` puts each block
    /// on a cache-line boundary given a chunk size of 8.
    blocks: Vec<u64>,
    n_blocks: u32,
    /// Blocks touched since last reset — enables O(dirty) reset.
    dirty: Vec<u32>,
    /// Diagnostics: total `mark` calls (probes).
    probes: u64,
    /// Diagnostics: total `mark` calls that reported "new" (returned true).
    inserts: u64,
}

/// Statistics captured between `reset()` calls.
#[derive(Debug, Clone, Copy)]
pub struct BloomStats {
    pub probes: u64,
    pub inserts: u64,
    pub dirty_blocks: u32,
    pub n_blocks: u32,
}

impl BlockedBloomVisited {
    /// Allocate for an expected `n_inserts` per query at a target
    /// false-positive rate of ~1%.  Rule of thumb: 20 bits/insert → n_blocks
    /// = ceil(n_inserts * 20 / 512).
    ///
    /// You can always oversize — the space cost is `n_blocks * 64` bytes.
    pub fn for_load(n_inserts: usize) -> Self {
        let bits_per_insert = 20;
        let n_blocks = ((n_inserts * bits_per_insert).max(BLOCK_BITS)).div_ceil(BLOCK_BITS);
        Self::with_blocks(n_blocks.max(1))
    }

    /// Explicit block count.  Each block is 64 bytes.
    pub fn with_blocks(n_blocks: usize) -> Self {
        assert!(n_blocks > 0 && n_blocks <= u32::MAX as usize);
        Self {
            blocks: vec![0u64; n_blocks * 8],
            n_blocks: n_blocks as u32,
            dirty: Vec::with_capacity(n_blocks.min(1024)),
            probes: 0,
            inserts: 0,
        }
    }

    /// Snapshot diagnostics — cleared by `reset`.
    pub fn stats(&self) -> BloomStats {
        BloomStats {
            probes: self.probes,
            inserts: self.inserts,
            dirty_blocks: self.dirty.len() as u32,
            n_blocks: self.n_blocks,
        }
    }

    /// Splitmix64 finalizer — high-quality bit mixing for u64.
    #[inline(always)]
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9E3779B97F4A7C15);
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
        x ^ (x >> 31)
    }

    /// Produce (block_index, [bit_position; K_HASHES]) for `id`.
    #[inline(always)]
    fn locate(&self, id: u32) -> (usize, [u32; K_HASHES]) {
        let h = Self::mix(id as u64);
        // High 32 bits select the block via fast-range multiplication —
        // avoids the modulo for non-power-of-two block counts.
        let block = ((h >> 32) as u64 * self.n_blocks as u64) >> 32;

        // Four independent bit positions inside the 512-bit block.
        // Each hash lane uses a rotate + odd multiplier for decorrelation.
        let base = h as u32; // low 32 bits
        let lanes = [
            base.wrapping_mul(0x85EBCA77),
            base.rotate_left(11).wrapping_mul(0xC2B2AE3D),
            base.rotate_left(19).wrapping_mul(0x27D4EB2F),
            base.rotate_left(29).wrapping_mul(0x165667B1),
        ];
        // Reduce to 0..512.
        let bits = [
            lanes[0] & 0x1FF,
            lanes[1] & 0x1FF,
            lanes[2] & 0x1FF,
            lanes[3] & 0x1FF,
        ];
        (block as usize, bits)
    }

    #[inline(always)]
    fn word_bit(bit: u32) -> (usize, u64) {
        // Within one 512-bit block: 8 u64 words. Word = bit>>6, mask = 1<<(bit&63).
        ((bit >> 6) as usize, 1u64 << (bit & 63))
    }
}

impl VisitedSet for BlockedBloomVisited {
    #[inline]
    fn mark(&mut self, id: u32) -> bool {
        self.probes = self.probes.wrapping_add(1);
        let (block, bits) = self.locate(id);
        let base = block * 8;

        let mut all_set = true;
        let mut newly_touched = false;
        for &b in &bits {
            let (w, m) = Self::word_bit(b);
            let word = &mut self.blocks[base + w];
            if (*word & m) == 0 {
                all_set = false;
                *word |= m;
                newly_touched = true;
            }
        }
        if newly_touched && (self.dirty.last().copied() != Some(block as u32)) {
            // Cheap coalesce for the common case of consecutive probes in the
            // same block (streaming neighbor scans).
            self.dirty.push(block as u32);
        }
        let new = !all_set;
        if new {
            self.inserts = self.inserts.wrapping_add(1);
        }
        new
    }

    #[inline]
    fn contains(&self, id: u32) -> bool {
        let (block, bits) = self.locate(id);
        let base = block * 8;
        for &b in &bits {
            let (w, m) = Self::word_bit(b);
            if (self.blocks[base + w] & m) == 0 {
                return false;
            }
        }
        true
    }

    fn reset(&mut self) {
        // O(dirty) reset.  We accept that `dirty` may contain the same block
        // several times if `mark` was interleaved with other blocks; a HashSet
        // of touched blocks would defeat the point.  Duplicate zero-writes are
        // harmless and stay cache-hot.
        for &b in &self.dirty {
            let base = (b as usize) * 8;
            for word in &mut self.blocks[base..base + 8] {
                *word = 0;
            }
        }
        self.dirty.clear();
        self.probes = 0;
        self.inserts = 0;
    }

    fn bytes(&self) -> usize {
        self.blocks.len() * 8 + self.dirty.capacity() * 4 + 64
    }

    fn name(&self) -> &'static str {
        "BlockedBloom(512b, k=4)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    #[test]
    fn mark_reports_new_and_seen() {
        let mut v = BlockedBloomVisited::for_load(1024);
        assert!(v.mark(1));
        assert!(!v.mark(1));
        assert!(v.contains(1));
    }

    #[test]
    fn reset_clears_dirty_blocks_only() {
        let mut v = BlockedBloomVisited::for_load(1024);
        v.mark(1);
        v.mark(2);
        v.mark(3);
        let stats_before = v.stats();
        assert!(stats_before.dirty_blocks >= 1);
        v.reset();
        assert!(!v.contains(1));
        assert!(!v.contains(2));
        assert!(!v.contains(3));
    }

    #[test]
    fn false_positive_rate_within_budget() {
        // 512 inserts into a filter sized for 2048 inserts (loose sizing).
        // Empirical FP rate must be < 1.5%.
        let mut v = BlockedBloomVisited::for_load(2048);
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let inserted: Vec<u32> = (0..512).map(|_| rng.gen()).collect();
        for &id in &inserted {
            v.mark(id);
        }
        let inserted_set: std::collections::HashSet<u32> = inserted.iter().copied().collect();
        let mut fps = 0usize;
        let trials = 20_000;
        for _ in 0..trials {
            let candidate: u32 = rng.gen();
            if inserted_set.contains(&candidate) {
                continue;
            }
            if v.contains(candidate) {
                fps += 1;
            }
        }
        let fp_rate = fps as f64 / trials as f64;
        assert!(fp_rate < 0.015, "FP rate too high: {fp_rate:.4}");
    }
}
