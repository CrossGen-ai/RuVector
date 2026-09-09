//! Neighbor-list storage backends.
//!
//! All backends operate on **sorted-ascending** neighbor lists. Decoding
//! writes into a caller-owned buffer to keep the hot path allocation-free.
//! The [`NeighborStore`] trait lets HNSW code choose an encoding without
//! recompilation-time coupling to a specific representation.

use crate::graph::Graph;

/// Common interface for neighbor storage.
pub trait NeighborStore {
    /// Number of nodes stored.
    fn len(&self) -> usize;
    /// True if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Total bytes owned by this store (heap payload only; ignores stack).
    fn bytes(&self) -> usize;
    /// Decode neighbors of `node` into `out` (cleared first).
    fn decode(&self, node: u32, out: &mut Vec<u32>);
}

// -------- FlatU32Store --------------------------------------------------

/// Baseline: contiguous `u32` arena + per-node offset table.
#[derive(Debug, Clone)]
pub struct FlatU32Store {
    data: Vec<u32>,
    offsets: Vec<u32>, // length = n + 1
}

impl FlatU32Store {
    /// Build from `graph`.
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.len();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut data: Vec<u32> = Vec::with_capacity(graph.total_edges());
        offsets.push(0);
        for row in &graph.adj {
            data.extend_from_slice(row);
            offsets.push(data.len() as u32);
        }
        Self { data, offsets }
    }
}

impl NeighborStore for FlatU32Store {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    fn bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u32>()
            + self.offsets.len() * std::mem::size_of::<u32>()
    }
    fn decode(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        let s = self.offsets[node as usize] as usize;
        let e = self.offsets[node as usize + 1] as usize;
        out.extend_from_slice(&self.data[s..e]);
    }
}

// -------- VarintDeltaStore ----------------------------------------------

/// Sorted-delta + LEB128 varint. Payload is byte-addressed; a per-node
/// offset table indexes into `data`.
#[derive(Debug, Clone)]
pub struct VarintDeltaStore {
    data: Vec<u8>,
    offsets: Vec<u32>, // byte offsets, length = n + 1
    lens: Vec<u16>,    // neighbor count per node (bounded by HNSW M<<65535)
}

fn encode_varint(mut x: u32, out: &mut Vec<u8>) {
    while x >= 0x80 {
        out.push(((x & 0x7f) as u8) | 0x80);
        x >>= 7;
    }
    out.push(x as u8);
}

#[inline]
fn decode_varint(bytes: &[u8], pos: &mut usize) -> u32 {
    let mut x: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let b = bytes[*pos];
        *pos += 1;
        x |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    x
}

impl VarintDeltaStore {
    /// Build from `graph`.
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.len();
        let mut data: Vec<u8> = Vec::new();
        let mut offsets = Vec::with_capacity(n + 1);
        let mut lens = Vec::with_capacity(n);
        offsets.push(0);
        for row in &graph.adj {
            let mut prev: i64 = -1;
            for &v in row {
                let delta = (v as i64 - prev) as u32;
                encode_varint(delta, &mut data);
                prev = v as i64;
            }
            offsets.push(data.len() as u32);
            lens.push(row.len() as u16);
        }
        Self { data, offsets, lens }
    }
}

impl NeighborStore for VarintDeltaStore {
    fn len(&self) -> usize {
        self.lens.len()
    }
    fn bytes(&self) -> usize {
        self.data.len()
            + self.offsets.len() * std::mem::size_of::<u32>()
            + self.lens.len() * std::mem::size_of::<u16>()
    }
    fn decode(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        let start = self.offsets[node as usize] as usize;
        let end = self.offsets[node as usize + 1] as usize;
        let bytes = &self.data[start..end];
        let mut pos = 0usize;
        let mut prev: i64 = -1;
        let k = self.lens[node as usize] as usize;
        out.reserve(k);
        for _ in 0..k {
            let d = decode_varint(bytes, &mut pos);
            let v = (prev + d as i64) as u32;
            out.push(v);
            prev = v as i64;
        }
    }
}

// -------- BitpackedDeltaStore -------------------------------------------

/// Sorted-delta + per-list fixed bit-width packing. For each node we store
/// the bit-width `w` (5 bits, so 1..=32) followed by `k*w` bits of deltas.
/// The `first` delta (from -1) uses the same width. This yields tighter
/// packing than varints when deltas are uniformly small (typical after
/// locality remapping).
#[derive(Debug, Clone)]
pub struct BitpackedDeltaStore {
    data: Vec<u8>,
    // packed layout: (bit_offset: u32, width: u8, len: u16) per node
    meta_bit_offset: Vec<u32>,
    meta_width: Vec<u8>,
    meta_len: Vec<u16>,
}

fn bit_write(buf: &mut Vec<u8>, bit_pos: &mut u64, value: u32, width: u8) {
    // Ensure capacity.
    let end_bit = *bit_pos + width as u64;
    let need_bytes = ((end_bit + 7) / 8) as usize;
    if buf.len() < need_bytes {
        buf.resize(need_bytes, 0);
    }
    let mut remaining = width as u32;
    let mut v = value;
    while remaining > 0 {
        let byte_idx = (*bit_pos / 8) as usize;
        let bit_in_byte = (*bit_pos % 8) as u32;
        let space = 8 - bit_in_byte;
        let take = remaining.min(space);
        let mask = if take == 32 { u32::MAX } else { (1u32 << take) - 1 };
        let chunk = (v & mask) as u8;
        buf[byte_idx] |= chunk << bit_in_byte;
        v >>= take;
        remaining -= take;
        *bit_pos += take as u64;
    }
}

#[inline]
fn bit_read(buf: &[u8], bit_pos: &mut u64, width: u8) -> u32 {
    let mut remaining = width as u32;
    let mut out: u32 = 0;
    let mut out_shift: u32 = 0;
    while remaining > 0 {
        let byte_idx = (*bit_pos / 8) as usize;
        let bit_in_byte = (*bit_pos % 8) as u32;
        let space = 8 - bit_in_byte;
        let take = remaining.min(space);
        let mask = if take == 32 { u32::MAX } else { (1u32 << take) - 1 };
        let byte = buf[byte_idx] as u32;
        let chunk = (byte >> bit_in_byte) & mask;
        out |= chunk << out_shift;
        out_shift += take;
        remaining -= take;
        *bit_pos += take as u64;
    }
    out
}

#[inline]
fn bits_for(x: u32) -> u8 {
    if x == 0 { 1 } else { (32 - x.leading_zeros()) as u8 }
}

impl BitpackedDeltaStore {
    /// Build from `graph`.
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.len();
        let mut data: Vec<u8> = Vec::new();
        let mut bit_offset: u64 = 0;
        let mut meta_bit_offset = Vec::with_capacity(n);
        let mut meta_width = Vec::with_capacity(n);
        let mut meta_len = Vec::with_capacity(n);
        for row in &graph.adj {
            // Determine max delta -> bit width.
            let mut prev: i64 = -1;
            let mut max_d: u32 = 0;
            for &v in row {
                let d = (v as i64 - prev) as u32;
                if d > max_d {
                    max_d = d;
                }
                prev = v as i64;
            }
            let w = bits_for(max_d.max(1)).max(1);
            meta_bit_offset.push(bit_offset as u32);
            meta_width.push(w);
            meta_len.push(row.len() as u16);
            let mut prev: i64 = -1;
            for &v in row {
                let d = (v as i64 - prev) as u32;
                bit_write(&mut data, &mut bit_offset, d, w);
                prev = v as i64;
            }
        }
        Self { data, meta_bit_offset, meta_width, meta_len }
    }
}

impl NeighborStore for BitpackedDeltaStore {
    fn len(&self) -> usize {
        self.meta_len.len()
    }
    fn bytes(&self) -> usize {
        self.data.len()
            + self.meta_bit_offset.len() * std::mem::size_of::<u32>()
            + self.meta_width.len() * std::mem::size_of::<u8>()
            + self.meta_len.len() * std::mem::size_of::<u16>()
    }
    fn decode(&self, node: u32, out: &mut Vec<u32>) {
        out.clear();
        let k = self.meta_len[node as usize] as usize;
        let w = self.meta_width[node as usize];
        let mut pos = self.meta_bit_offset[node as usize] as u64;
        let mut prev: i64 = -1;
        out.reserve(k);
        for _ in 0..k {
            let d = bit_read(&self.data, &mut pos, w);
            let v = (prev + d as i64) as u32;
            out.push(v);
            prev = v as i64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{brute_knn_graph, random_vectors};
    use crate::remap::{apply_permutation, locality_remap_bfs};

    fn assert_roundtrip<S: NeighborStore>(store: &S, graph: &Graph) {
        let mut buf = Vec::with_capacity(64);
        for i in 0..graph.len() {
            store.decode(i as u32, &mut buf);
            assert_eq!(&buf[..], graph.adj[i].as_slice(), "mismatch at {}", i);
        }
    }

    fn build_graph() -> Graph {
        let v = random_vectors(200, 16, 11);
        let g = brute_knn_graph(&v, 200, 16, 12);
        let perm = locality_remap_bfs(&g, 0);
        apply_permutation(&g, &perm)
    }

    #[test]
    fn flat_roundtrip() {
        let g = build_graph();
        let s = FlatU32Store::from_graph(&g);
        assert_roundtrip(&s, &g);
    }

    #[test]
    fn varint_roundtrip() {
        let g = build_graph();
        let s = VarintDeltaStore::from_graph(&g);
        assert_roundtrip(&s, &g);
    }

    #[test]
    fn bitpacked_roundtrip() {
        let g = build_graph();
        let s = BitpackedDeltaStore::from_graph(&g);
        assert_roundtrip(&s, &g);
    }

    #[test]
    fn encoded_stores_smaller_than_flat_after_remap() {
        let g = build_graph();
        let flat = FlatU32Store::from_graph(&g).bytes();
        let varint = VarintDeltaStore::from_graph(&g).bytes();
        let bp = BitpackedDeltaStore::from_graph(&g).bytes();
        assert!(varint < flat, "varint {} !< flat {}", varint, flat);
        assert!(bp < flat, "bitpacked {} !< flat {}", bp, flat);
    }
}
