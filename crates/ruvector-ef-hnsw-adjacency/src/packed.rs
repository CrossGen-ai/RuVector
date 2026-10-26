//! Fixed-width bit-packed baseline: every neighbor id is stored in
//! `ceil(log2(n))` bits in a single contiguous bitstream. This is a stronger
//! baseline than `PlainAdjacency` because for `n = 100k` the width drops from
//! 32 to 17 bits per id — a ~47 % saving before any clever coding.

use crate::Adjacency;

#[derive(Debug)]
pub struct PackedAdjacency {
    offsets: Vec<u32>,
    data: Vec<u64>,
    bits_per_id: u8,
    total_ids: usize,
}

impl PackedAdjacency {
    pub fn new(universe: usize) -> Self {
        let bits_per_id = bits_needed(universe.max(2) as u64);
        Self {
            offsets: vec![0],
            data: Vec::new(),
            bits_per_id,
            total_ids: 0,
        }
    }

    #[inline]
    fn ensure_capacity(&mut self, extra_bits: usize) {
        let needed_words = ((self.total_ids * self.bits_per_id as usize) + extra_bits + 63) / 64;
        if needed_words > self.data.len() {
            self.data.resize(needed_words, 0);
        }
    }
}

#[inline]
fn bits_needed(max_val: u64) -> u8 {
    64 - max_val.leading_zeros() as u8
}

impl Adjacency for PackedAdjacency {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn push(&mut self, neighbors: &[u32]) {
        let mut tmp = neighbors.to_vec();
        tmp.sort_unstable();
        let bits = self.bits_per_id as usize;
        self.ensure_capacity(tmp.len() * bits);
        for &id in &tmp {
            let bit_offset = self.total_ids * bits;
            write_bits(&mut self.data, bit_offset, bits, id as u64);
            self.total_ids += 1;
        }
        self.offsets.push(self.total_ids as u32);
    }

    fn bytes(&self) -> usize {
        self.offsets.len() * std::mem::size_of::<u32>()
            + self.data.len() * std::mem::size_of::<u64>()
    }

    fn decode_into(&self, node: usize, out: &mut Vec<u32>) {
        out.clear();
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        let bits = self.bits_per_id as usize;
        for i in start..end {
            let val = read_bits(&self.data, i * bits, bits);
            out.push(val as u32);
        }
    }
}

#[inline]
fn write_bits(data: &mut [u64], bit_offset: usize, bits: usize, value: u64) {
    debug_assert!(bits > 0 && bits <= 64);
    let word = bit_offset / 64;
    let shift = bit_offset % 64;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let v = value & mask;
    data[word] |= v << shift;
    if shift + bits > 64 {
        data[word + 1] |= v >> (64 - shift);
    }
}

#[inline]
fn read_bits(data: &[u64], bit_offset: usize, bits: usize) -> u64 {
    debug_assert!(bits > 0 && bits <= 64);
    let word = bit_offset / 64;
    let shift = bit_offset % 64;
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let low = data[word] >> shift;
    let combined = if shift + bits > 64 {
        low | (data[word + 1] << (64 - shift))
    } else {
        low
    };
    combined & mask
}
