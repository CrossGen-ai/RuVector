//! Bit-packed HNSW neighbor-list compression.
//!
//! Provides three swappable backends behind [`NeighborStore`]:
//!
//! * [`RawU32Store`]  — baseline `Vec<u32>` layout used by hnswlib/faiss.
//! * [`DeltaVarintStore`] — sort neighbors, delta-encode, varint-pack.
//! * [`BitPackedStore`]   — sort neighbors, delta-encode, fixed-width bit-pack
//!   with per-list bit-width header (SIMD-BP-style).
//!
//! Design goals:
//!
//! 1. **Random-access decode of a single node's adjacency** — no whole-index
//!    scan.
//! 2. **Deterministic layout** — the same graph re-encodes to the same bytes.
//! 3. **Pure safe Rust, no unsafe, no deps** (keeps the PoC surface tiny).
//!
//! All three backends must produce the same *set* of neighbors when decoded.
//! `DeltaVarintStore` and `BitPackedStore` return neighbors *sorted ascending*
//! (delta encoding requires ordering). Callers that need HNSW's insertion
//! order should keep that ordering in a separate side buffer, or accept sorted
//! order — for graph traversal it does not matter.

#![deny(unsafe_code)]

use std::io::{self, Read, Write};

/// Common interface every graph-compression backend implements.
pub trait NeighborStore: Sized {
    /// Encode adjacency lists.  `lists[i]` is node `i`'s neighbors.
    fn build(lists: &[Vec<u32>]) -> Self;
    /// Decode neighbors of node `id` into `out` (cleared first).
    fn decode(&self, id: u32, out: &mut Vec<u32>);
    /// Total bytes on the wire (payload + index).
    fn bytes(&self) -> usize;
    /// Number of nodes.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// 1. Raw u32 baseline (mirrors hnswlib/faiss on-disk layout).
// ---------------------------------------------------------------------------

/// `Vec<u32>` per node, prefixed with a u32 length.  Zero-compression baseline.
#[derive(Debug)]
pub struct RawU32Store {
    /// Flattened `[u32]` payload.  Each list is [len, id0, id1, ...].
    payload: Vec<u32>,
    /// Offset (in u32 words) into `payload` for each node.
    offsets: Vec<u32>,
}

impl NeighborStore for RawU32Store {
    fn build(lists: &[Vec<u32>]) -> Self {
        let mut payload: Vec<u32> = Vec::new();
        let mut offsets: Vec<u32> = Vec::with_capacity(lists.len() + 1);
        for l in lists {
            offsets.push(payload.len() as u32);
            payload.push(l.len() as u32);
            payload.extend_from_slice(l);
        }
        offsets.push(payload.len() as u32);
        RawU32Store { payload, offsets }
    }

    fn decode(&self, id: u32, out: &mut Vec<u32>) {
        out.clear();
        let start = self.offsets[id as usize] as usize;
        let n = self.payload[start] as usize;
        out.extend_from_slice(&self.payload[start + 1..start + 1 + n]);
    }

    fn bytes(&self) -> usize {
        self.payload.len() * 4 + self.offsets.len() * 4
    }

    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

// ---------------------------------------------------------------------------
// 2. Delta + varint (Google's group-varint / protobuf-style).
// ---------------------------------------------------------------------------

/// Sort each list, delta-encode successive ids, LEB128-varint-pack.
#[derive(Debug)]
pub struct DeltaVarintStore {
    payload: Vec<u8>,
    /// Byte offset of node `i`'s list.
    offsets: Vec<u32>,
    /// Number of neighbors per node.
    lens: Vec<u16>,
}

fn write_varint(buf: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        buf.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

fn read_varint(data: &[u8], pos: &mut usize) -> u32 {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let b = data[*pos];
        *pos += 1;
        result |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            return result;
        }
        shift += 7;
    }
}

impl NeighborStore for DeltaVarintStore {
    fn build(lists: &[Vec<u32>]) -> Self {
        let mut payload = Vec::new();
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        let mut lens = Vec::with_capacity(lists.len());
        let mut scratch = Vec::<u32>::new();
        for l in lists {
            offsets.push(payload.len() as u32);
            lens.push(l.len() as u16);
            scratch.clear();
            scratch.extend_from_slice(l);
            scratch.sort_unstable();
            let mut prev = 0u32;
            for &id in &scratch {
                let delta = id.wrapping_sub(prev);
                write_varint(&mut payload, delta);
                prev = id;
            }
        }
        offsets.push(payload.len() as u32);
        DeltaVarintStore {
            payload,
            offsets,
            lens,
        }
    }

    fn decode(&self, id: u32, out: &mut Vec<u32>) {
        out.clear();
        let n = self.lens[id as usize] as usize;
        let mut pos = self.offsets[id as usize] as usize;
        let mut prev = 0u32;
        for _ in 0..n {
            let d = read_varint(&self.payload, &mut pos);
            prev = prev.wrapping_add(d);
            out.push(prev);
        }
    }

    fn bytes(&self) -> usize {
        self.payload.len() + self.offsets.len() * 4 + self.lens.len() * 2
    }

    fn len(&self) -> usize {
        self.lens.len()
    }
}

// ---------------------------------------------------------------------------
// 3. Fixed-width bit-packing with per-list bit-width header.
// ---------------------------------------------------------------------------

/// Per-list: sort → delta → find the max delta → pick the minimal `bits`
/// such that all deltas fit in `bits` bits.  Store `bits` + packed payload.
#[derive(Debug)]
pub struct BitPackedStore {
    payload: Vec<u8>,
    offsets: Vec<u32>,
    lens: Vec<u16>,
    bits_per_list: Vec<u8>,
}

fn pack_bits(buf: &mut Vec<u8>, values: &[u32], bits: u8) {
    if bits == 0 {
        return;
    }
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mask: u64 = if bits == 64 { !0u64 } else { (1u64 << bits) - 1 };
    for &v in values {
        acc |= ((v as u64) & mask) << acc_bits;
        acc_bits += bits as u32;
        while acc_bits >= 8 {
            buf.push((acc & 0xFF) as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        buf.push((acc & 0xFF) as u8);
    }
}

fn unpack_bits(data: &[u8], start_byte: usize, bits: u8, count: usize, out: &mut Vec<u32>) {
    if bits == 0 {
        for _ in 0..count {
            out.push(0);
        }
        return;
    }
    let mask: u64 = if bits == 64 { !0u64 } else { (1u64 << bits) - 1 };
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut byte_pos = start_byte;
    for _ in 0..count {
        while acc_bits < bits as u32 {
            acc |= (data[byte_pos] as u64) << acc_bits;
            byte_pos += 1;
            acc_bits += 8;
        }
        out.push((acc & mask) as u32);
        acc >>= bits;
        acc_bits -= bits as u32;
    }
}

fn min_bits_for(max_val: u32) -> u8 {
    if max_val == 0 {
        0
    } else {
        32 - max_val.leading_zeros() as u8
    }
}

impl NeighborStore for BitPackedStore {
    fn build(lists: &[Vec<u32>]) -> Self {
        let mut payload = Vec::new();
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        let mut lens = Vec::with_capacity(lists.len());
        let mut bits_per_list = Vec::with_capacity(lists.len());
        let mut scratch = Vec::<u32>::new();
        let mut deltas = Vec::<u32>::new();
        for l in lists {
            offsets.push(payload.len() as u32);
            lens.push(l.len() as u16);
            scratch.clear();
            scratch.extend_from_slice(l);
            scratch.sort_unstable();
            deltas.clear();
            let mut prev = 0u32;
            let mut max_d = 0u32;
            for &id in &scratch {
                let d = id.wrapping_sub(prev);
                if d > max_d {
                    max_d = d;
                }
                deltas.push(d);
                prev = id;
            }
            let bits = min_bits_for(max_d);
            bits_per_list.push(bits);
            pack_bits(&mut payload, &deltas, bits);
        }
        offsets.push(payload.len() as u32);
        BitPackedStore {
            payload,
            offsets,
            lens,
            bits_per_list,
        }
    }

    fn decode(&self, id: u32, out: &mut Vec<u32>) {
        out.clear();
        let n = self.lens[id as usize] as usize;
        let bits = self.bits_per_list[id as usize];
        let start = self.offsets[id as usize] as usize;
        let mut tmp = Vec::with_capacity(n);
        unpack_bits(&self.payload, start, bits, n, &mut tmp);
        let mut prev = 0u32;
        for d in tmp {
            prev = prev.wrapping_add(d);
            out.push(prev);
        }
    }

    fn bytes(&self) -> usize {
        self.payload.len()
            + self.offsets.len() * 4
            + self.lens.len() * 2
            + self.bits_per_list.len()
    }

    fn len(&self) -> usize {
        self.lens.len()
    }
}

// ---------------------------------------------------------------------------
// Snapshot I/O (for a swappable, on-disk format later).
// ---------------------------------------------------------------------------

/// Very small snapshot format: only for `BitPackedStore`, primarily so ADR
/// consumers can point to a serialized artifact.
pub fn write_bitpacked<W: Write>(store: &BitPackedStore, mut w: W) -> io::Result<usize> {
    let mut n = 0;
    w.write_all(&(store.lens.len() as u32).to_le_bytes())?;
    n += 4;
    for &len in &store.lens {
        w.write_all(&len.to_le_bytes())?;
        n += 2;
    }
    for &b in &store.bits_per_list {
        w.write_all(&[b])?;
        n += 1;
    }
    w.write_all(&(store.offsets.len() as u32).to_le_bytes())?;
    n += 4;
    for &o in &store.offsets {
        w.write_all(&o.to_le_bytes())?;
        n += 4;
    }
    w.write_all(&(store.payload.len() as u32).to_le_bytes())?;
    n += 4;
    w.write_all(&store.payload)?;
    n += store.payload.len();
    Ok(n)
}

pub fn read_bitpacked<R: Read>(mut r: R) -> io::Result<BitPackedStore> {
    let mut buf4 = [0u8; 4];
    r.read_exact(&mut buf4)?;
    let n_lists = u32::from_le_bytes(buf4) as usize;
    let mut lens = Vec::with_capacity(n_lists);
    for _ in 0..n_lists {
        let mut b2 = [0u8; 2];
        r.read_exact(&mut b2)?;
        lens.push(u16::from_le_bytes(b2));
    }
    let mut bits_per_list = vec![0u8; n_lists];
    r.read_exact(&mut bits_per_list)?;
    r.read_exact(&mut buf4)?;
    let n_off = u32::from_le_bytes(buf4) as usize;
    let mut offsets = Vec::with_capacity(n_off);
    for _ in 0..n_off {
        r.read_exact(&mut buf4)?;
        offsets.push(u32::from_le_bytes(buf4));
    }
    r.read_exact(&mut buf4)?;
    let payload_len = u32::from_le_bytes(buf4) as usize;
    let mut payload = vec![0u8; payload_len];
    r.read_exact(&mut payload)?;
    Ok(BitPackedStore {
        payload,
        offsets,
        lens,
        bits_per_list,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PCG-ish RNG so tests are reproducible without deps.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            s
        }
    }

    fn synth_graph(n_nodes: usize, degree: usize, seed: u64) -> Vec<Vec<u32>> {
        let mut r = rng(seed);
        let mut out = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            let mut l = Vec::with_capacity(degree);
            for _ in 0..degree {
                l.push((r() as u32) % (n_nodes as u32));
            }
            out.push(l);
        }
        out
    }

    fn sorted(v: &[u32]) -> Vec<u32> {
        let mut x = v.to_vec();
        x.sort_unstable();
        x
    }

    #[test]
    fn raw_roundtrip() {
        let g = synth_graph(500, 16, 42);
        let s = RawU32Store::build(&g);
        let mut out = Vec::new();
        for i in 0..g.len() {
            s.decode(i as u32, &mut out);
            assert_eq!(out, g[i]);
        }
    }

    #[test]
    fn varint_roundtrip() {
        let g = synth_graph(500, 16, 7);
        let s = DeltaVarintStore::build(&g);
        let mut out = Vec::new();
        for i in 0..g.len() {
            s.decode(i as u32, &mut out);
            assert_eq!(out, sorted(&g[i]));
        }
    }

    #[test]
    fn bitpacked_roundtrip() {
        let g = synth_graph(500, 16, 11);
        let s = BitPackedStore::build(&g);
        let mut out = Vec::new();
        for i in 0..g.len() {
            s.decode(i as u32, &mut out);
            assert_eq!(out, sorted(&g[i]));
        }
    }

    #[test]
    fn snapshot_roundtrip() {
        let g = synth_graph(300, 12, 3);
        let s = BitPackedStore::build(&g);
        let mut buf = Vec::new();
        let n_written = write_bitpacked(&s, &mut buf).unwrap();
        assert_eq!(n_written, buf.len());
        let s2 = read_bitpacked(&buf[..]).unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..g.len() {
            s.decode(i as u32, &mut a);
            s2.decode(i as u32, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn compression_beats_raw() {
        // On a realistic HNSW-like graph, both compressors must beat the u32
        // baseline.  4096 nodes × M=16 is representative.
        let g = synth_graph(4096, 16, 99);
        let raw = RawU32Store::build(&g);
        let vb = DeltaVarintStore::build(&g);
        let bp = BitPackedStore::build(&g);
        assert!(vb.bytes() < raw.bytes(), "varint {} vs raw {}", vb.bytes(), raw.bytes());
        assert!(bp.bytes() < raw.bytes(), "bitpack {} vs raw {}", bp.bytes(), raw.bytes());
    }

    #[test]
    fn min_bits_edges() {
        assert_eq!(min_bits_for(0), 0);
        assert_eq!(min_bits_for(1), 1);
        assert_eq!(min_bits_for(7), 3);
        assert_eq!(min_bits_for(u32::MAX), 32);
    }
}
