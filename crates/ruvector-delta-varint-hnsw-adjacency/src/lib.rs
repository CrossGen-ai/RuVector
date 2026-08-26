//! Delta+Varint and PFor-blocked adjacency compression for HNSW graphs.
//!
//! ## Motivation
//!
//! In HNSW, each node stores a fixed-M list of neighbor IDs per layer. For a
//! 100 M-vector index with M=32 at layer 0, the adjacency alone consumes
//! 100e6 * 32 * 4 B = **12.8 GiB** as raw `u32`s — often larger than the
//! quantized vector payload itself (e.g. PQ-64 = 6.4 GiB).
//!
//! Neighbor lists in real graphs exhibit locality when nodes are laid out via
//! space-filling curves, k-means partitioning, or recursive graph bisection.
//! Sorting each list and delta-encoding shrinks it by 2–4x, and per-block
//! bit-packing (PFor) recovers most of the entropy while keeping decode
//! branch-free.
//!
//! ## Design
//!
//! Three interchangeable backends behind [`AdjacencyStore`]:
//!
//! 1. [`PlainAdjacency`] — baseline, raw `u32` array.
//! 2. [`DeltaVarintAdjacency`] — sort + first-id-absolute + delta + LEB128 varint.
//! 3. [`PforBlockedAdjacency`] — sort + delta + fixed-width bit-packed blocks
//!    with a per-node header. Branch-free block decode.
//!
//! ## Guarantees
//!
//! - Lossless: `decode(encode(xs)) == sort(xs)`. Order of neighbors is not
//!   preserved (HNSW does not require it — only the set matters for search).
//! - `#![forbid(unsafe_code)]`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use core::fmt;

/// Storage of per-node neighbor ID lists for a single HNSW layer.
///
/// Implementations are expected to be immutable snapshots produced by
/// [`AdjacencyBuilder`]. Concurrent search uses `&self` only.
pub trait AdjacencyStore: Send + Sync {
    /// Number of nodes.
    fn len(&self) -> usize;

    /// True when there are no nodes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total bytes owned by the store (payload, not `Vec` capacity slack).
    fn bytes(&self) -> usize;

    /// Decode neighbors of `node` into `out`. Returns the number decoded.
    ///
    /// `out` must be at least `max_degree()` long.
    fn decode_into(&self, node: u32, out: &mut [u32]) -> usize;

    /// Upper bound on any node's degree.
    fn max_degree(&self) -> usize;
}

/// Builder helper: normalize a neighbor list (sort + dedup).
fn normalize(neighbors: &[u32]) -> Vec<u32> {
    let mut v: Vec<u32> = neighbors.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

// -------------------------------------------------------------------------
// PlainAdjacency
// -------------------------------------------------------------------------

/// Baseline: raw `u32` array with fixed slot width.
///
/// Unused slots are filled with `u32::MAX` sentinel.
pub struct PlainAdjacency {
    m: usize,
    slots: Vec<u32>,
    lens: Vec<u8>,
}

impl PlainAdjacency {
    /// Build from per-node neighbor lists (any order).
    pub fn build<I>(nodes: I, m: usize) -> Self
    where
        I: IntoIterator<Item = Vec<u32>>,
    {
        assert!(m <= 255, "M must fit in u8");
        let mut slots = Vec::new();
        let mut lens = Vec::new();
        for ns in nodes {
            let ns = normalize(&ns);
            let take = ns.len().min(m);
            lens.push(take as u8);
            slots.extend_from_slice(&ns[..take]);
            for _ in take..m {
                slots.push(u32::MAX);
            }
        }
        Self { m, slots, lens }
    }
}

impl AdjacencyStore for PlainAdjacency {
    fn len(&self) -> usize {
        self.lens.len()
    }
    fn bytes(&self) -> usize {
        self.slots.len() * 4 + self.lens.len()
    }
    fn max_degree(&self) -> usize {
        self.m
    }
    fn decode_into(&self, node: u32, out: &mut [u32]) -> usize {
        let i = node as usize;
        let n = self.lens[i] as usize;
        let base = i * self.m;
        out[..n].copy_from_slice(&self.slots[base..base + n]);
        n
    }
}

// -------------------------------------------------------------------------
// DeltaVarintAdjacency
// -------------------------------------------------------------------------

/// Sort + first-id-absolute + delta + LEB128 varint.
///
/// Layout per node in `stream`:
/// `[len:u8] [id0 varint] [d1 varint] .. [d(len-1) varint]`
/// where `d_k = id_k - id_{k-1}`.
pub struct DeltaVarintAdjacency {
    m: usize,
    stream: Vec<u8>,
    offsets: Vec<u32>, // start offset per node; extra sentinel at end.
}

impl DeltaVarintAdjacency {
    /// Build from per-node neighbor lists.
    pub fn build<I>(nodes: I, m: usize) -> Self
    where
        I: IntoIterator<Item = Vec<u32>>,
    {
        assert!(m <= 255, "M must fit in u8");
        let nodes: Vec<Vec<u32>> = nodes.into_iter().collect();
        let mut stream = Vec::with_capacity(nodes.len() * m);
        let mut offsets = Vec::with_capacity(nodes.len() + 1);
        for ns in &nodes {
            offsets.push(stream.len() as u32);
            let ns = normalize(ns);
            let take = ns.len().min(m);
            stream.push(take as u8);
            if take == 0 {
                continue;
            }
            write_varint(&mut stream, ns[0]);
            for w in ns[..take].windows(2) {
                let d = w[1] - w[0];
                write_varint(&mut stream, d);
            }
        }
        offsets.push(stream.len() as u32);
        Self { m, stream, offsets }
    }
}

impl AdjacencyStore for DeltaVarintAdjacency {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    fn bytes(&self) -> usize {
        self.stream.len() + self.offsets.len() * 4
    }
    fn max_degree(&self) -> usize {
        self.m
    }
    fn decode_into(&self, node: u32, out: &mut [u32]) -> usize {
        let i = node as usize;
        let mut p = self.offsets[i] as usize;
        let n = self.stream[p] as usize;
        p += 1;
        if n == 0 {
            return 0;
        }
        let (v0, adv) = read_varint(&self.stream[p..]);
        p += adv;
        out[0] = v0;
        let mut prev = v0;
        for k in 1..n {
            let (d, adv) = read_varint(&self.stream[p..]);
            p += adv;
            prev += d;
            out[k] = prev;
        }
        n
    }
}

// -------------------------------------------------------------------------
// PforBlockedAdjacency
// -------------------------------------------------------------------------

/// Sort + delta + fixed-width bit-packed blocks.
///
/// For each node we store:
/// `[len:u8] [bitwidth:u8] [id0:u32-LE] [packed deltas of (len-1) values]`
///
/// The bitwidth is the minimum number of bits needed for the max delta.
/// Decode is branch-free and vectorization-friendly.
pub struct PforBlockedAdjacency {
    m: usize,
    stream: Vec<u8>,
    offsets: Vec<u32>,
}

impl PforBlockedAdjacency {
    /// Build from per-node neighbor lists.
    pub fn build<I>(nodes: I, m: usize) -> Self
    where
        I: IntoIterator<Item = Vec<u32>>,
    {
        assert!(m <= 255, "M must fit in u8");
        let nodes: Vec<Vec<u32>> = nodes.into_iter().collect();
        let mut stream = Vec::new();
        let mut offsets = Vec::with_capacity(nodes.len() + 1);
        for ns in &nodes {
            offsets.push(stream.len() as u32);
            let ns = normalize(ns);
            let take = ns.len().min(m);
            if take == 0 {
                stream.push(0u8);
                stream.push(0u8);
                continue;
            }
            let deltas: Vec<u32> = ns[..take]
                .windows(2)
                .map(|w| w[1] - w[0])
                .collect();
            let max_d = deltas.iter().copied().max().unwrap_or(0);
            let bw: u8 = if max_d == 0 { 0 } else { 32 - max_d.leading_zeros() as u8 };
            stream.push(take as u8);
            stream.push(bw);
            stream.extend_from_slice(&ns[0].to_le_bytes());
            if bw > 0 {
                pack_bits(&deltas, bw, &mut stream);
            }
        }
        offsets.push(stream.len() as u32);
        Self { m, stream, offsets }
    }
}

impl AdjacencyStore for PforBlockedAdjacency {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    fn bytes(&self) -> usize {
        self.stream.len() + self.offsets.len() * 4
    }
    fn max_degree(&self) -> usize {
        self.m
    }
    fn decode_into(&self, node: u32, out: &mut [u32]) -> usize {
        let i = node as usize;
        let mut p = self.offsets[i] as usize;
        let n = self.stream[p] as usize;
        let bw = self.stream[p + 1] as usize;
        p += 2;
        if n == 0 {
            return 0;
        }
        let id0 = u32::from_le_bytes([
            self.stream[p],
            self.stream[p + 1],
            self.stream[p + 2],
            self.stream[p + 3],
        ]);
        p += 4;
        out[0] = id0;
        if n == 1 {
            return 1;
        }
        let count = n - 1;
        let mut prev = id0;
        if bw == 0 {
            // All deltas are zero — impossible after dedup unless count==0,
            // but be safe.
            for k in 1..n {
                out[k] = prev;
            }
            return n;
        }
        let mut buf = [0u32; 256];
        unpack_bits(&self.stream[p..], bw as u8, count, &mut buf);
        for k in 0..count {
            prev += buf[k];
            out[k + 1] = prev;
        }
        n
    }
}

// -------------------------------------------------------------------------
// Varint codec
// -------------------------------------------------------------------------

fn write_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn read_varint(buf: &[u8]) -> (u32, usize) {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    let mut i: usize = 0;
    loop {
        let b = buf[i];
        i += 1;
        result |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            return (result, i);
        }
        shift += 7;
        if shift >= 32 {
            return (result, i);
        }
    }
}

// -------------------------------------------------------------------------
// Bit-packing (fixed width per node)
// -------------------------------------------------------------------------

fn pack_bits(values: &[u32], bw: u8, out: &mut Vec<u8>) {
    let bw = bw as u32;
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    for &v in values {
        acc |= (v as u64) << acc_bits;
        acc_bits += bw;
        while acc_bits >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            acc_bits -= 8;
        }
    }
    if acc_bits > 0 {
        out.push(acc as u8);
    }
}

fn unpack_bits(buf: &[u8], bw: u8, count: usize, out: &mut [u32]) {
    let bw = bw as u32;
    let mask: u64 = if bw == 32 { u32::MAX as u64 } else { (1u64 << bw) - 1 };
    let mut acc: u64 = 0;
    let mut acc_bits: u32 = 0;
    let mut bi: usize = 0;
    for k in 0..count {
        while acc_bits < bw {
            acc |= (buf[bi] as u64) << acc_bits;
            bi += 1;
            acc_bits += 8;
        }
        out[k] = (acc & mask) as u32;
        acc >>= bw;
        acc_bits -= bw;
    }
}

// -------------------------------------------------------------------------
// Convenience: bytes-per-node breakdown
// -------------------------------------------------------------------------

/// Human-readable footprint report.
#[derive(Debug, Clone)]
pub struct Footprint {
    /// Name of the backend.
    pub name: &'static str,
    /// Total bytes.
    pub bytes: usize,
    /// Number of nodes.
    pub nodes: usize,
    /// Bytes per node.
    pub bytes_per_node: f64,
}

impl fmt::Display for Footprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:>28}: {:>10} B total, {:>7.2} B/node",
            self.name, self.bytes, self.bytes_per_node
        )
    }
}

/// Report footprint for any store.
pub fn footprint(name: &'static str, store: &dyn AdjacencyStore) -> Footprint {
    let bytes = store.bytes();
    let nodes = store.len();
    Footprint {
        name,
        bytes,
        nodes,
        bytes_per_node: if nodes == 0 { 0.0 } else { bytes as f64 / nodes as f64 },
    }
}

// -------------------------------------------------------------------------
// Deterministic HNSW-style neighbor generator (for tests + benches)
// -------------------------------------------------------------------------

/// Deterministic RNG (xorshift64) — no external deps.
pub struct Xorshift(pub u64);

impl Xorshift {
    /// Next `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Next `u32` in `[0, n)`.
    pub fn gen_range(&mut self, n: u32) -> u32 {
        (self.next_u64() % (n as u64)) as u32
    }
}

/// Generate synthetic HNSW-like neighbor lists.
///
/// `locality` controls how tight the neighbors are around each node id.
/// `locality=1.0` yields near-linear locality (best case for delta coding);
/// `locality=0.0` yields uniform random over `[0, n)`.
pub fn synth_neighbors(n: usize, m: usize, locality: f32, seed: u64) -> Vec<Vec<u32>> {
    let mut rng = Xorshift(seed);
    let n_u = n as u32;
    let window = ((n as f32) * (1.0 - locality.clamp(0.0, 1.0))).max(m as f32 * 4.0) as u32;
    let mut out = Vec::with_capacity(n);
    for i in 0..n_u {
        let mut ns = Vec::with_capacity(m);
        for _ in 0..m {
            let lo = i.saturating_sub(window / 2);
            let hi = (i + window / 2 + 1).min(n_u);
            let range = hi - lo;
            let v = lo + rng.gen_range(range);
            if v != i {
                ns.push(v);
            }
        }
        out.push(ns);
    }
    out
}
