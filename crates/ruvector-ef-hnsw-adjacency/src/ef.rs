//! Elias-Fano per-node adjacency encoding.
//!
//! For a sorted neighbor list `x_0 <= x_1 <= ... <= x_{m-1}` with values in
//! `[0, u)`, Elias-Fano stores:
//!   * `low[i] = x_i & ((1 << L) - 1)` — fixed-width `L` bits per value
//!   * `high[i] = x_i >> L` — written as `high[i] - high[i-1]` zeros followed
//!     by a single `1` bit (differential unary)
//!
//! With `L = max(0, floor(log2(u / m)))`, the total cost is `m*L + m +
//! max_high` bits ≈ `m * (2 + log2(u/m))` on average, versus `m * ceil(log2
//! u)` for a packed baseline.
//!
//! The implementation below keeps every node's low/high streams in two
//! shared bitvectors with per-node offsets so a 1-M-node HNSW layer costs
//! `O(n)` extra `u32` offsets rather than a `Vec` allocation per node.

use crate::Adjacency;

#[derive(Debug)]
pub struct EFAdjacency {
    universe: u64,
    l_bits: u8,
    low_offsets: Vec<u64>,  // bit offsets, length n+1
    high_offsets: Vec<u64>, // bit offsets, length n+1
    counts: Vec<u32>,       // neighbors per node
    low_data: Vec<u64>,
    high_data: Vec<u64>,
    low_bits_total: u64,
    high_bits_total: u64,
}

impl EFAdjacency {
    pub fn new(universe: usize) -> Self {
        let u = universe.max(2) as u64;
        // L chosen per-instance from the universe; nodes vary in m so a
        // tiny m stays efficient and a large m degrades gracefully.
        // For HNSW with M ≈ sqrt(u), log2(u / M) ≈ log2(u) / 2.
        let avg_m: u64 = 16; // conservative default for HNSW layer 0.
        let ratio = u / avg_m.max(1);
        let l_bits = if ratio <= 1 {
            0
        } else {
            63 - ratio.leading_zeros() as u8
        };
        Self {
            universe: u,
            l_bits,
            low_offsets: vec![0],
            high_offsets: vec![0],
            counts: Vec::new(),
            low_data: Vec::new(),
            high_data: Vec::new(),
            low_bits_total: 0,
            high_bits_total: 0,
        }
    }

    #[inline]
    fn ensure_low_capacity(&mut self, extra_bits: u64) {
        let needed = ((self.low_bits_total + extra_bits + 63) / 64) as usize;
        if needed > self.low_data.len() {
            self.low_data.resize(needed, 0);
        }
    }

    #[inline]
    fn ensure_high_capacity(&mut self, extra_bits: u64) {
        let needed = ((self.high_bits_total + extra_bits + 63) / 64) as usize;
        if needed > self.high_data.len() {
            self.high_data.resize(needed, 0);
        }
    }
}

impl Adjacency for EFAdjacency {
    fn len(&self) -> usize {
        self.counts.len()
    }

    fn push(&mut self, neighbors: &[u32]) {
        let mut tmp = neighbors.to_vec();
        tmp.sort_unstable();
        let m = tmp.len() as u32;
        let l = self.l_bits as usize;
        let low_mask = if l == 0 { 0 } else { (1u64 << l) - 1 };

        // --- LOW BITS ---
        let low_bits_needed = tmp.len() as u64 * l as u64;
        self.ensure_low_capacity(low_bits_needed);
        if l > 0 {
            for &id in &tmp {
                let v = (id as u64) & low_mask;
                write_bits(&mut self.low_data, self.low_bits_total, l, v);
                self.low_bits_total += l as u64;
            }
        }

        // --- HIGH BITS (differential unary) ---
        let mut prev_high: u64 = 0;
        let max_high = if tmp.is_empty() {
            0
        } else {
            (*tmp.last().unwrap() as u64) >> l
        };
        let high_bits_needed = tmp.len() as u64 + max_high;
        self.ensure_high_capacity(high_bits_needed);
        for &id in &tmp {
            let hi = (id as u64) >> l;
            let zeros = hi - prev_high;
            // Write `zeros` zero bits then a single `1`.
            self.high_bits_total += zeros;
            // Set the `1` bit at the current position, then advance.
            set_bit(&mut self.high_data, self.high_bits_total);
            self.high_bits_total += 1;
            prev_high = hi;
        }

        self.counts.push(m);
        self.low_offsets.push(self.low_bits_total);
        self.high_offsets.push(self.high_bits_total);
    }

    fn bytes(&self) -> usize {
        self.low_offsets.len() * std::mem::size_of::<u64>()
            + self.high_offsets.len() * std::mem::size_of::<u64>()
            + self.counts.len() * std::mem::size_of::<u32>()
            + self.low_data.len() * std::mem::size_of::<u64>()
            + self.high_data.len() * std::mem::size_of::<u64>()
    }

    fn decode_into(&self, node: usize, out: &mut Vec<u32>) {
        out.clear();
        let m = self.counts[node] as usize;
        if m == 0 {
            return;
        }
        let l = self.l_bits as usize;
        let low_start = self.low_offsets[node];
        let high_start = self.high_offsets[node];
        let high_end = self.high_offsets[node + 1];

        let mut hi: u64 = 0;
        let mut pos = high_start;
        let mut emitted = 0;
        // Scan the high bitstream: every 0 means "increment high component",
        // every 1 means "emit the current pair".
        while pos < high_end && emitted < m {
            if get_bit(&self.high_data, pos) {
                let low_val = if l == 0 {
                    0
                } else {
                    read_bits(&self.low_data, low_start + (emitted as u64) * (l as u64), l)
                };
                let value = (hi << l) | low_val;
                out.push(value as u32);
                emitted += 1;
            } else {
                hi += 1;
            }
            pos += 1;
        }
        debug_assert_eq!(emitted, m);
        let _ = self.universe; // silence dead-field warning in release builds.
    }
}

#[inline]
fn write_bits(data: &mut [u64], bit_offset: u64, bits: usize, value: u64) {
    debug_assert!(bits > 0 && bits <= 64);
    let word = (bit_offset / 64) as usize;
    let shift = (bit_offset % 64) as u32;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let v = value & mask;
    data[word] |= v << shift;
    if shift as usize + bits > 64 {
        data[word + 1] |= v >> (64 - shift as usize);
    }
}

#[inline]
fn read_bits(data: &[u64], bit_offset: u64, bits: usize) -> u64 {
    debug_assert!(bits > 0 && bits <= 64);
    let word = (bit_offset / 64) as usize;
    let shift = (bit_offset % 64) as u32;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let low = data[word] >> shift;
    let combined = if shift as usize + bits > 64 {
        low | (data[word + 1] << (64 - shift as usize))
    } else {
        low
    };
    combined & mask
}

#[inline]
fn set_bit(data: &mut [u64], bit_offset: u64) {
    let word = (bit_offset / 64) as usize;
    let shift = (bit_offset % 64) as u32;
    data[word] |= 1u64 << shift;
}

#[inline]
fn get_bit(data: &[u64], bit_offset: u64) -> bool {
    let word = (bit_offset / 64) as usize;
    let shift = (bit_offset % 64) as u32;
    (data[word] >> shift) & 1 == 1
}
