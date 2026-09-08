//! Swappable adjacency-store backends.
//!
//! Each backend implements [`Adjacency`], which exposes:
//!   * `len()` — number of nodes.
//!   * `neighbors_into(v, out)` — decode neighbour list for node `v` into
//!     `out` (reused between calls, avoiding per-query allocation).
//!   * `bytes()` — self-reported *payload* memory usage in bytes. This is
//!     what we compare in benchmarks. Payload only: excludes Rust `Vec`
//!     capacity slack and the 24 B header per outer Vec, unless the
//!     backend needs those slots to function (see `DenseAdj`).
//!
//! Two encoded backends share the same varbyte codec (see private
//! helpers) so numbers are apples-to-apples.

use crate::NodeId;

/// Adjacency-store trait. All graph mutation happens through
/// [`from_lists`] — encoded backends are built once from a plain
/// `Vec<Vec<NodeId>>` produced by the graph builder.
pub trait Adjacency: Send + Sync {
    fn from_lists(lists: Vec<Vec<NodeId>>) -> Self
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn neighbors_into(&self, v: NodeId, out: &mut Vec<NodeId>);
    /// Payload bytes reported by the encoding. Used in benchmarks.
    fn bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Backend 1: plain Vec<Vec<u32>>. This is the baseline every alternative
// must beat on memory to be worth considering.
// ---------------------------------------------------------------------------

pub struct DenseAdj {
    lists: Vec<Vec<NodeId>>,
}

impl Adjacency for DenseAdj {
    fn from_lists(lists: Vec<Vec<NodeId>>) -> Self {
        Self { lists }
    }
    fn len(&self) -> usize {
        self.lists.len()
    }
    fn neighbors_into(&self, v: NodeId, out: &mut Vec<NodeId>) {
        out.clear();
        out.extend_from_slice(&self.lists[v as usize]);
    }
    fn bytes(&self) -> usize {
        // Outer Vec header + per-list header + per-neighbour u32.
        // We charge the outer header because a real deployment must keep
        // it — the alternative encodings replace this with a flat blob.
        let outer_hdr = std::mem::size_of::<Vec<NodeId>>() * self.lists.len();
        let inner: usize = self
            .lists
            .iter()
            .map(|l| l.len() * std::mem::size_of::<NodeId>())
            .sum();
        outer_hdr + inner
    }
    fn name(&self) -> &'static str {
        "dense_vec_vec_u32"
    }
}

// ---------------------------------------------------------------------------
// VarByte codec.
//
// Unsigned LEB128-style: 7 bits per byte, MSB=1 => more follows.
// - values 0..=127        → 1 byte
// - values 128..=16_383   → 2 bytes
// - values ..=2_097_151   → 3 bytes  (covers 2M-node graphs of any density)
// - values ..=268_435_455 → 4 bytes
// - larger                → 5 bytes  (u32 max)
// ---------------------------------------------------------------------------

fn vb_write(mut x: u32, out: &mut Vec<u8>) {
    while x >= 0x80 {
        out.push(((x & 0x7F) as u8) | 0x80);
        x >>= 7;
    }
    out.push(x as u8);
}

/// Decode a single varint starting at `buf[pos]`. Returns (value, new_pos).
#[inline]
fn vb_read(buf: &[u8], mut pos: usize) -> (u32, usize) {
    let mut x: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let b = buf[pos];
        pos += 1;
        x |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            return (x, pos);
        }
        shift += 7;
    }
}

// ---------------------------------------------------------------------------
// Backend 2: delta + varbyte over the *original* node ids.
// Layout:
//   blob: [ vb(len_v0) vb(d0) vb(d1) ... | vb(len_v1) ... | ... ]
//   offsets: Vec<u32> of length N+1 into blob.
// ---------------------------------------------------------------------------

pub struct DeltaVarByteAdj {
    n: usize,
    blob: Vec<u8>,
    offsets: Vec<u32>,
}

fn encode_lists(lists: &[Vec<NodeId>]) -> (Vec<u8>, Vec<u32>) {
    let mut blob = Vec::with_capacity(lists.len() * 8);
    let mut offsets = Vec::with_capacity(lists.len() + 1);
    for list in lists {
        offsets.push(blob.len() as u32);
        // Sort a copy so deltas are strictly increasing and small.
        let mut sorted = list.clone();
        sorted.sort_unstable();
        vb_write(sorted.len() as u32, &mut blob);
        let mut prev: u32 = 0;
        for &n in &sorted {
            // First entry stored as raw value (delta from 0).
            vb_write(n - prev, &mut blob);
            prev = n;
        }
    }
    offsets.push(blob.len() as u32);
    (blob, offsets)
}

fn decode_at(blob: &[u8], start: u32, end: u32, out: &mut Vec<NodeId>) {
    out.clear();
    let start = start as usize;
    let end = end as usize;
    if start == end {
        return;
    }
    let (len, mut pos) = vb_read(blob, start);
    let mut prev: u32 = 0;
    for _ in 0..len {
        let (d, np) = vb_read(blob, pos);
        pos = np;
        prev += d;
        out.push(prev);
        if pos >= end {
            break;
        }
    }
}

impl Adjacency for DeltaVarByteAdj {
    fn from_lists(lists: Vec<Vec<NodeId>>) -> Self {
        let (blob, offsets) = encode_lists(&lists);
        Self {
            n: lists.len(),
            blob,
            offsets,
        }
    }
    fn len(&self) -> usize {
        self.n
    }
    fn neighbors_into(&self, v: NodeId, out: &mut Vec<NodeId>) {
        let s = self.offsets[v as usize];
        let e = self.offsets[v as usize + 1];
        decode_at(&self.blob, s, e, out);
    }
    fn bytes(&self) -> usize {
        self.blob.len() + self.offsets.len() * std::mem::size_of::<u32>()
    }
    fn name(&self) -> &'static str {
        "delta_varbyte"
    }
}

// ---------------------------------------------------------------------------
// Backend 3: BFS-reorder node ids so that graph neighbours have similar
// ids, THEN delta + varbyte. The re-permutation happens once at build time;
// benchmarks charge decode cost only.
//
// External API preserves original ids: `neighbors_into(v_orig, ...)` yields
// original ids. Internally we translate through the permutation.
// ---------------------------------------------------------------------------

pub struct ReorderedDeltaAdj {
    n: usize,
    blob: Vec<u8>,
    offsets: Vec<u32>,
    /// old_id -> new_id (position in reordered space)
    old_to_new: Vec<NodeId>,
    /// new_id -> old_id (inverse)
    new_to_old: Vec<NodeId>,
}

impl ReorderedDeltaAdj {
    /// Build from lists plus a precomputed permutation.
    pub fn from_lists_with_permutation(
        lists: Vec<Vec<NodeId>>,
        old_to_new: Vec<NodeId>,
    ) -> Self {
        let n = lists.len();
        assert_eq!(old_to_new.len(), n);
        let mut new_to_old = vec![0u32; n];
        for (old, &new_id) in old_to_new.iter().enumerate() {
            new_to_old[new_id as usize] = old as u32;
        }
        // Re-index each neighbour list into new-id space and store in
        // new-id order (so linear scans over ids are cache-friendly).
        let mut relabelled: Vec<Vec<NodeId>> = vec![Vec::new(); n];
        for old in 0..n {
            let new_id = old_to_new[old] as usize;
            let l = &lists[old];
            let mut mapped: Vec<NodeId> =
                l.iter().map(|&x| old_to_new[x as usize]).collect();
            mapped.sort_unstable();
            relabelled[new_id] = mapped;
        }
        let (blob, offsets) = encode_lists(&relabelled);
        Self {
            n,
            blob,
            offsets,
            old_to_new,
            new_to_old,
        }
    }
}

impl Adjacency for ReorderedDeltaAdj {
    fn from_lists(lists: Vec<Vec<NodeId>>) -> Self {
        // Fallback when no permutation supplied: identity. Callers should
        // use `from_lists_with_permutation` for the real benefit.
        let n = lists.len();
        let identity: Vec<NodeId> = (0..n as NodeId).collect();
        Self::from_lists_with_permutation(lists, identity)
    }
    fn len(&self) -> usize {
        self.n
    }
    fn neighbors_into(&self, v_old: NodeId, out: &mut Vec<NodeId>) {
        let v_new = self.old_to_new[v_old as usize];
        let s = self.offsets[v_new as usize];
        let e = self.offsets[v_new as usize + 1];
        decode_at(&self.blob, s, e, out);
        // Translate back to old ids so the graph search sees a consistent
        // id space.
        for x in out.iter_mut() {
            *x = self.new_to_old[*x as usize];
        }
    }
    fn bytes(&self) -> usize {
        self.blob.len()
            + self.offsets.len() * std::mem::size_of::<u32>()
            + self.old_to_new.len() * std::mem::size_of::<NodeId>()
            + self.new_to_old.len() * std::mem::size_of::<NodeId>()
    }
    fn name(&self) -> &'static str {
        "reordered_delta_varbyte"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varbyte_roundtrip_boundaries() {
        for &v in &[0u32, 1, 127, 128, 16_383, 16_384, 2_097_151, u32::MAX] {
            let mut buf = Vec::new();
            vb_write(v, &mut buf);
            let (dec, _) = vb_read(&buf, 0);
            assert_eq!(v, dec);
        }
    }

    #[test]
    fn delta_roundtrip_matches_dense() {
        let lists: Vec<Vec<NodeId>> = vec![
            vec![5, 1, 3, 9],
            vec![],
            vec![7],
            vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        ];
        let dense = DenseAdj::from_lists(lists.clone());
        let enc = DeltaVarByteAdj::from_lists(lists);
        for v in 0..dense.len() as u32 {
            let mut a = Vec::new();
            let mut b = Vec::new();
            dense.neighbors_into(v, &mut a);
            enc.neighbors_into(v, &mut b);
            a.sort_unstable();
            assert_eq!(a, b);
        }
    }
}
