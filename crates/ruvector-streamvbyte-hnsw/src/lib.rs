//! StreamVByte delta encoding for HNSW adjacency lists.
//!
//! Three codecs implemented behind a common `AdjacencyCodec` trait:
//!
//! - `PlainCodec`     : baseline, 4 bytes per neighbor ID (u32 LE).
//! - `DeltaVarintCodec`: zigzag/delta + LEB128-style varint (branchy decode).
//! - `StreamVByteCodec`: StreamVByte (Lemire 2017) — control bytes packed
//!                       separately so the data stream decodes with a
//!                       table-driven shuffle. We implement the scalar form;
//!                       it is still branch-light and cache-friendly, and
//!                       maps cleanly onto SSSE3 `pshufb` on x86_64.
//!
//! Keys numbers reported by the benches live in the research doc.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::io::{self, Read, Write};

pub trait AdjacencyCodec {
    fn name(&self) -> &'static str;
    /// Encode a sorted-ascending neighbor list into `out`.
    fn encode(&self, neighbors: &[u32], out: &mut Vec<u8>);
    /// Decode a previously-encoded list of exactly `len` ids into `out`.
    fn decode(&self, bytes: &[u8], len: usize, out: &mut Vec<u32>);
}

// ---------------------------------------------------------------------------
// Plain u32 LE
// ---------------------------------------------------------------------------

pub struct PlainCodec;

impl AdjacencyCodec for PlainCodec {
    fn name(&self) -> &'static str { "plain-u32" }

    fn encode(&self, neighbors: &[u32], out: &mut Vec<u8>) {
        out.reserve(neighbors.len() * 4);
        for &n in neighbors {
            out.extend_from_slice(&n.to_le_bytes());
        }
    }

    fn decode(&self, bytes: &[u8], len: usize, out: &mut Vec<u32>) {
        out.clear();
        out.reserve(len);
        for i in 0..len {
            let s = i * 4;
            let v = u32::from_le_bytes([bytes[s], bytes[s + 1], bytes[s + 2], bytes[s + 3]]);
            out.push(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Delta + LEB128 varint
// ---------------------------------------------------------------------------

pub struct DeltaVarintCodec;

fn write_varint(mut v: u32, w: &mut Vec<u8>) {
    while v >= 0x80 {
        w.push((v as u8) | 0x80);
        v >>= 7;
    }
    w.push(v as u8);
}

fn read_varint(bytes: &[u8], pos: &mut usize) -> u32 {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let b = bytes[*pos];
        *pos += 1;
        result |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 { return result; }
        shift += 7;
    }
}

impl AdjacencyCodec for DeltaVarintCodec {
    fn name(&self) -> &'static str { "delta-varint" }

    fn encode(&self, neighbors: &[u32], out: &mut Vec<u8>) {
        let mut prev = 0u32;
        for &n in neighbors {
            let delta = n.wrapping_sub(prev);
            write_varint(delta, out);
            prev = n;
        }
    }

    fn decode(&self, bytes: &[u8], len: usize, out: &mut Vec<u32>) {
        out.clear();
        out.reserve(len);
        let mut pos = 0usize;
        let mut prev = 0u32;
        for _ in 0..len {
            let delta = read_varint(bytes, &mut pos);
            prev = prev.wrapping_add(delta);
            out.push(prev);
        }
    }
}

// ---------------------------------------------------------------------------
// StreamVByte (delta variant, scalar). Group of 4 deltas → 1 control byte
// (2 bits per delta: 00=1B, 01=2B, 10=3B, 11=4B) + packed data.
// ---------------------------------------------------------------------------

pub struct StreamVByteCodec;

#[inline]
fn width_code(v: u32) -> u8 {
    // Branchless: compute floor(log2(v)/8) via bit tricks.
    //   v==0 -> 1 byte
    //   1..=0xff -> 1 byte -> code 0
    //   0x100..=0xffff -> code 1
    //   0x10000..=0xffffff -> code 2
    //   else code 3
    if v < 0x100 { 0 }
    else if v < 0x1_0000 { 1 }
    else if v < 0x100_0000 { 2 }
    else { 3 }
}

#[inline]
fn write_width(code: u8, v: u32, data: &mut Vec<u8>) {
    let bytes = v.to_le_bytes();
    // (code+1) bytes written.
    data.extend_from_slice(&bytes[..=code as usize]);
}

#[inline]
fn read_width(code: u8, bytes: &[u8], pos: &mut usize) -> u32 {
    let n = (code + 1) as usize;
    let mut buf = [0u8; 4];
    buf[..n].copy_from_slice(&bytes[*pos..*pos + n]);
    *pos += n;
    u32::from_le_bytes(buf)
}

impl AdjacencyCodec for StreamVByteCodec {
    fn name(&self) -> &'static str { "stream-vbyte-delta" }

    fn encode(&self, neighbors: &[u32], out: &mut Vec<u8>) {
        let n = neighbors.len();
        let num_groups = (n + 3) / 4;
        // Reserve control-byte block first.
        let ctrl_start = out.len();
        out.resize(ctrl_start + num_groups, 0u8);

        let mut prev = 0u32;
        let mut gi = 0usize;
        let mut i = 0usize;
        while i < n {
            let mut ctrl = 0u8;
            let group_end = (i + 4).min(n);
            // Emit each delta into the data stream; collect its 2-bit code.
            for lane in 0..(group_end - i) {
                let v = neighbors[i + lane];
                let d = v.wrapping_sub(prev);
                prev = v;
                let c = width_code(d);
                ctrl |= c << (lane * 2);
                write_width(c, d, out);
            }
            // Partial group: pad control with zero (no data written for pad lanes).
            out[ctrl_start + gi] = ctrl;
            gi += 1;
            i = group_end;
        }
    }

    fn decode(&self, bytes: &[u8], len: usize, out: &mut Vec<u32>) {
        out.clear();
        out.reserve(len);
        let num_groups = (len + 3) / 4;
        let mut pos = num_groups; // data starts after control bytes
        let mut prev = 0u32;
        for g in 0..num_groups {
            let ctrl = bytes[g];
            let lanes = if g + 1 == num_groups {
                len - g * 4
            } else { 4 };
            for lane in 0..lanes {
                let code = (ctrl >> (lane * 2)) & 0b11;
                let delta = read_width(code, bytes, &mut pos);
                prev = prev.wrapping_add(delta);
                out.push(prev);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HNSW adjacency store
// ---------------------------------------------------------------------------

/// Compressed adjacency for one HNSW layer.
///
/// Simple columnar layout: `offsets[i]..offsets[i+1]` is node `i`'s encoded
/// neighbor block; `lens[i]` is the neighbor count (needed because codecs
/// do not length-prefix). Keeps the data path linear and cache-friendly.
pub struct CompressedAdjacency<C: AdjacencyCodec> {
    pub codec: C,
    pub offsets: Vec<u32>,
    pub lens: Vec<u16>,
    pub data: Vec<u8>,
}

impl<C: AdjacencyCodec> CompressedAdjacency<C> {
    pub fn build(codec: C, lists: &[Vec<u32>]) -> Self {
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        let mut lens = Vec::with_capacity(lists.len());
        let mut data = Vec::with_capacity(lists.len() * 16);
        offsets.push(0);
        for list in lists {
            let mut sorted = list.clone();
            sorted.sort_unstable();
            codec.encode(&sorted, &mut data);
            offsets.push(data.len() as u32);
            lens.push(sorted.len() as u16);
        }
        Self { codec, offsets, lens, data }
    }

    pub fn neighbors(&self, node: usize, scratch: &mut Vec<u32>) {
        let s = self.offsets[node] as usize;
        let e = self.offsets[node + 1] as usize;
        self.codec.decode(&self.data[s..e], self.lens[node] as usize, scratch);
    }

    pub fn total_bytes(&self) -> usize {
        self.data.len() + self.offsets.len() * 4 + self.lens.len() * 2
    }
}

// ---------------------------------------------------------------------------
// IO helpers for a crude on-disk dump; used by the demo binary.
// ---------------------------------------------------------------------------

pub fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn random_sorted(n: usize, max: u32, seed: u64) -> Vec<u32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut v: Vec<u32> = (0..n).map(|_| rng.gen_range(0..max)).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    #[test]
    fn plain_roundtrip() {
        let input = random_sorted(32, 1_000_000, 1);
        let codec = PlainCodec;
        let mut buf = Vec::new();
        codec.encode(&input, &mut buf);
        let mut out = Vec::new();
        codec.decode(&buf, input.len(), &mut out);
        assert_eq!(input, out);
    }

    #[test]
    fn varint_roundtrip() {
        let input = random_sorted(64, 10_000_000, 2);
        let codec = DeltaVarintCodec;
        let mut buf = Vec::new();
        codec.encode(&input, &mut buf);
        let mut out = Vec::new();
        codec.decode(&buf, input.len(), &mut out);
        assert_eq!(input, out);
    }

    #[test]
    fn svb_roundtrip_small() {
        // Also exercises the partial final group.
        for n in [1usize, 2, 3, 4, 5, 6, 7, 8, 17, 33] {
            let input = random_sorted(n, 10_000_000, 100 + n as u64);
            let codec = StreamVByteCodec;
            let mut buf = Vec::new();
            codec.encode(&input, &mut buf);
            let mut out = Vec::new();
            codec.decode(&buf, input.len(), &mut out);
            assert_eq!(input, out, "n={}", n);
        }
    }

    #[test]
    fn svb_roundtrip_large() {
        let input = random_sorted(2000, 50_000_000, 7);
        let codec = StreamVByteCodec;
        let mut buf = Vec::new();
        codec.encode(&input, &mut buf);
        let mut out = Vec::new();
        codec.decode(&buf, input.len(), &mut out);
        assert_eq!(input, out);
    }

    #[test]
    fn adjacency_roundtrip() {
        // Simulate a tiny HNSW: 100 nodes, degree 16, ids in [0, 1_000_000).
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let lists: Vec<Vec<u32>> = (0..100)
            .map(|_| {
                let mut v: Vec<u32> = (0..16).map(|_| rng.gen_range(0..1_000_000u32)).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();

        let plain = CompressedAdjacency::build(PlainCodec, &lists);
        let svb = CompressedAdjacency::build(StreamVByteCodec, &lists);
        assert!(
            svb.total_bytes() < plain.total_bytes(),
            "SVB {} should be smaller than plain {}",
            svb.total_bytes(), plain.total_bytes()
        );

        let mut scratch = Vec::new();
        for (i, list) in lists.iter().enumerate() {
            let mut expected = list.clone();
            expected.sort_unstable();
            svb.neighbors(i, &mut scratch);
            assert_eq!(scratch, expected);
        }
    }
}
