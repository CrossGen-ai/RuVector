//! Backend 3: EliasFanoStore — one EF-encoded sequence per node, all
//! packed into two shared bit vectors (low + high) with per-node
//! offsets. See crate root for the trait contract.

use crate::NeighborStore;

#[derive(Debug, Default)]
struct BitVec {
    words: Vec<u64>,
    n_bits: usize,
}

impl BitVec {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            n_bits: 0,
        }
    }
    fn push_bit(&mut self, b: bool) {
        let w = self.n_bits >> 6;
        let bit = self.n_bits & 63;
        if w >= self.words.len() {
            self.words.push(0);
        }
        if b {
            self.words[w] |= 1u64 << bit;
        }
        self.n_bits += 1;
    }
    /// Ensure the underlying `Vec<u64>` has capacity for `extra_bits`
    /// additional bits without further reallocation. Called by
    /// [`EliasFanoStore::push_list`] to bound Vec-growth waste.
    fn reserve_bits(&mut self, extra_bits: usize) {
        let needed_words = (self.n_bits + extra_bits + 63) >> 6;
        if needed_words > self.words.capacity() {
            self.words.reserve(needed_words - self.words.len());
        }
    }
    fn push_bits(&mut self, v: u64, width: u32) {
        for i in 0..width {
            self.push_bit(((v >> i) & 1) == 1);
        }
    }
    fn get_bit(&self, i: usize) -> bool {
        ((self.words[i >> 6] >> (i & 63)) & 1) == 1
    }
    fn get_bits(&self, start: usize, width: u32) -> u64 {
        if width == 0 {
            return 0;
        }
        let mut v: u64 = 0;
        for i in 0..width {
            if self.get_bit(start + i as usize) {
                v |= 1u64 << i;
            }
        }
        v
    }
    fn bytes(&self) -> usize {
        self.words.capacity() * 8
    }
}

/// Per-node header for the Elias-Fano store.
#[derive(Debug, Clone, Copy)]
struct EfHeader {
    n: u32,        // count of neighbors
    l_width: u8,   // low bits per value
    low_off: u64,  // start bit offset in low vec
    high_off: u64, // start bit offset in high vec
}

#[derive(Debug, Default)]
pub struct EliasFanoStore {
    low: BitVec,
    high: BitVec,
    headers: Vec<EfHeader>,
}

impl EliasFanoStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            low: BitVec::new(),
            high: BitVec::new(),
            headers: Vec::new(),
        }
    }

    /// Number of low bits for `n` values with maximum `max`.
    fn compute_l(n: u32, max: u32) -> u8 {
        if n == 0 || max == 0 {
            return 0;
        }
        // Elias-Fano optimal `l = max(0, floor(log2(U/n)))` where U = max+1.
        let u = u64::from(max) + 1;
        let ratio = u / u64::from(n);
        if ratio <= 1 {
            0
        } else {
            (63 - ratio.leading_zeros()) as u8
        }
    }
}

impl NeighborStore for EliasFanoStore {
    fn push_list(&mut self, neighbors: &[u32]) {
        let mut v = neighbors.to_vec();
        v.sort_unstable();
        let n = v.len() as u32;
        let max = *v.last().unwrap_or(&0);
        let l_width = Self::compute_l(n, max);
        let l_mask: u64 = if l_width == 0 {
            0
        } else {
            (1u64 << l_width) - 1
        };
        let header = EfHeader {
            n,
            l_width,
            low_off: self.low.n_bits as u64,
            high_off: self.high.n_bits as u64,
        };
        // Reserve exactly the bits we're about to write to avoid the
        // Vec-of-u64 doubling behavior wasting up to 2x space.
        self.low.reserve_bits(n as usize * l_width as usize);
        let high_bits_needed = if l_width == 0 {
            u64::from(max) as usize + n as usize
        } else {
            (u64::from(max) >> l_width) as usize + n as usize
        };
        self.high.reserve_bits(high_bits_needed);
        // low: pack low `l` bits of each value in order.
        // high: unary code the high parts: for each i in 0..n write
        // `(high_i - high_{i-1})` zero bits then a 1. Equivalently:
        // set bit at position `high_i + i`.
        let mut prev_high: u64 = 0;
        for (i, x) in v.iter().enumerate() {
            let xv = u64::from(*x);
            let low = xv & l_mask;
            let high = xv >> l_width;
            self.low.push_bits(low, u32::from(l_width));
            // gap zeros then a 1
            let gap = high - prev_high;
            for _ in 0..gap {
                self.high.push_bit(false);
            }
            self.high.push_bit(true);
            prev_high = high;
            let _ = i;
        }
        self.headers.push(header);
    }

    fn len(&self) -> usize {
        self.headers.len()
    }

    fn get_list(&self, node_id: usize, out: &mut Vec<u32>) {
        out.clear();
        let h = self.headers[node_id];
        if h.n == 0 {
            return;
        }
        let l_width = h.l_width;
        // Walk the high bit-vector: for each 1 bit, its high part is
        // (position - popcount_of_ones_so_far).
        let mut ones_seen: u64 = 0;
        let mut pos = h.high_off as usize;
        let mut low_pos = h.low_off as usize;
        while ones_seen < u64::from(h.n) {
            if self.high.get_bit(pos) {
                let high = (pos as u64 - h.high_off) - ones_seen;
                let low = self.low.get_bits(low_pos, u32::from(l_width));
                let v = (high << l_width) | low;
                out.push(v as u32);
                low_pos += l_width as usize;
                ones_seen += 1;
            }
            pos += 1;
        }
    }

    fn nth(&self, node_id: usize, k: usize) -> u32 {
        // Scan for the (k+1)-th set bit in high, then combine with
        // the k-th low chunk. O(k) but with a very tight inner loop —
        // realistic HNSW M is small (16-64).
        let h = self.headers[node_id];
        let l_width = h.l_width;
        let mut ones_seen: u64 = 0;
        let mut pos = h.high_off as usize;
        let low_pos = h.low_off as usize + k * l_width as usize;
        loop {
            if self.high.get_bit(pos) {
                if ones_seen == k as u64 {
                    let high = (pos as u64 - h.high_off) - ones_seen;
                    let low = self.low.get_bits(low_pos, u32::from(l_width));
                    return ((high << l_width) | low) as u32;
                }
                ones_seen += 1;
            }
            pos += 1;
        }
    }

    fn list_len(&self, node_id: usize) -> usize {
        self.headers[node_id].n as usize
    }

    fn bytes(&self) -> usize {
        self.low.bytes()
            + self.high.bytes()
            + self.headers.capacity() * std::mem::size_of::<EfHeader>()
    }
}
