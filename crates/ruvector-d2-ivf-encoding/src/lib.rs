//! D2-IVF: Delta-of-Delta encoding for IVF posting-list IDs.
//!
//! An IVF posting list is a Vec<u32> of row IDs for one cluster. After a
//! locality reorder (ivf-centroid-locality-reorder + a per-cluster sort by
//! Hilbert key of the vector) the IDs within a posting are approximately
//! uniformly increasing: a near-arithmetic progression. Classical PFOR/varint
//! delta coding wins over raw u32 here; delta-of-delta (D2), borrowed from
//! time-series DBs (Gorilla/InfluxDB), wins further because second-differences
//! of a near-arithmetic sequence are near zero, letting us store most IDs in
//! 1-3 bits via a small zigzag/varint scheme.
//!
//! This crate provides three encoders behind a common trait so a probe
//! kernel can swap backends:
//!   * `RawU32`   : baseline, 4 bytes/ID
//!   * `VarintDelta` : single-delta zigzag-varint (PFOR-lite)
//!   * `D2Varint` : delta-of-delta zigzag-varint (this nightly)
//!
//! No `unsafe`, no SIMD intrinsics, no allocator tricks — the geometry win
//! is what we measure. The `bin/bench.rs` binary produces the numbers that
//! land in the research doc.

use std::io::{Cursor, Read};

/// Pluggable posting-list codec.
pub trait PostingCodec {
    fn name(&self) -> &'static str;
    fn encode(&self, ids: &[u32]) -> Vec<u8>;
    fn decode(&self, bytes: &[u8]) -> Vec<u32>;
}

/// ----------------------------------------------------------------------
/// Raw u32 baseline.
/// ----------------------------------------------------------------------
pub struct RawU32;

impl PostingCodec for RawU32 {
    fn name(&self) -> &'static str { "raw-u32" }

    fn encode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out
    }

    fn decode(&self, bytes: &[u8]) -> Vec<u32> {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let b = &bytes[i * 4..i * 4 + 4];
            out.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        out
    }
}

/// ----------------------------------------------------------------------
/// Zigzag + LEB128 varint helpers. Signed ints fold to u64 via zigzag,
/// then LEB128 emits 7 bits / byte with a continuation MSB.
/// ----------------------------------------------------------------------

#[inline]
fn zz_encode(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

#[inline]
fn zz_decode(u: u64) -> i64 {
    ((u >> 1) as i64) ^ -((u & 1) as i64)
}

fn varint_write(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(((v as u8) & 0x7F) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn varint_read(cur: &mut Cursor<&[u8]>) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut buf = [0u8; 1];
    loop {
        if cur.read_exact(&mut buf).is_err() {
            return None;
        }
        let b = buf[0];
        result |= ((b & 0x7F) as u64) << shift;
        if b < 0x80 {
            return Some(result);
        }
        shift += 7;
        if shift > 63 {
            return None;
        }
    }
}

/// ----------------------------------------------------------------------
/// VarintDelta: first ID as zz-varint, remaining as single-delta zz-varint.
/// Standard PFOR-lite baseline.
/// ----------------------------------------------------------------------
pub struct VarintDelta;

impl PostingCodec for VarintDelta {
    fn name(&self) -> &'static str { "varint-delta" }

    fn encode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 2);
        varint_write(&mut out, ids.len() as u64);
        if ids.is_empty() {
            return out;
        }
        varint_write(&mut out, zz_encode(ids[0] as i64));
        for w in ids.windows(2) {
            let d = w[1] as i64 - w[0] as i64;
            varint_write(&mut out, zz_encode(d));
        }
        out
    }

    fn decode(&self, bytes: &[u8]) -> Vec<u32> {
        let mut cur = Cursor::new(bytes);
        let n = varint_read(&mut cur).unwrap_or(0) as usize;
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return out;
        }
        let mut cur_id = zz_decode(varint_read(&mut cur).unwrap_or(0));
        out.push(cur_id as u32);
        for _ in 1..n {
            let d = zz_decode(varint_read(&mut cur).unwrap_or(0));
            cur_id += d;
            out.push(cur_id as u32);
        }
        out
    }
}

/// ----------------------------------------------------------------------
/// D2Varint: first ID + first delta, then zz-varint of delta-of-delta.
/// For approximately-arithmetic sequences the DoDs cluster at 0, so each
/// emits a single byte (one varint byte = zz(0) = 0x00).
/// ----------------------------------------------------------------------
pub struct D2Varint;

impl PostingCodec for D2Varint {
    fn name(&self) -> &'static str { "d2-varint" }

    fn encode(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len());
        varint_write(&mut out, ids.len() as u64);
        if ids.is_empty() {
            return out;
        }
        varint_write(&mut out, zz_encode(ids[0] as i64));
        if ids.len() == 1 {
            return out;
        }
        let first_delta = ids[1] as i64 - ids[0] as i64;
        varint_write(&mut out, zz_encode(first_delta));
        let mut prev_delta = first_delta;
        for i in 2..ids.len() {
            let d = ids[i] as i64 - ids[i - 1] as i64;
            let dod = d - prev_delta;
            varint_write(&mut out, zz_encode(dod));
            prev_delta = d;
        }
        out
    }

    fn decode(&self, bytes: &[u8]) -> Vec<u32> {
        let mut cur = Cursor::new(bytes);
        let n = varint_read(&mut cur).unwrap_or(0) as usize;
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return out;
        }
        let mut cur_id = zz_decode(varint_read(&mut cur).unwrap_or(0));
        out.push(cur_id as u32);
        if n == 1 {
            return out;
        }
        let mut prev_delta = zz_decode(varint_read(&mut cur).unwrap_or(0));
        cur_id += prev_delta;
        out.push(cur_id as u32);
        for _ in 2..n {
            let dod = zz_decode(varint_read(&mut cur).unwrap_or(0));
            let d = prev_delta + dod;
            cur_id += d;
            out.push(cur_id as u32);
            prev_delta = d;
        }
        out
    }
}

/// ----------------------------------------------------------------------
/// Synthetic IVF posting generator.
/// Produces `num_clusters` posting lists summing to `n_total` IDs,
/// where IDs within each posting are ordered by a 1-D "Hilbert key"
/// surrogate (sorted after assignment) to mimic a locality reorder.
/// ----------------------------------------------------------------------
pub fn synth_postings(
    n_total: usize,
    num_clusters: usize,
    seed: u64,
) -> Vec<Vec<u32>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut assign = vec![0u32; n_total];
    for a in assign.iter_mut() {
        *a = rng.gen_range(0..num_clusters as u32);
    }
    let mut postings: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
    for (id, &c) in assign.iter().enumerate() {
        postings[c as usize].push(id as u32);
    }
    // Already sorted by construction (IDs appended in increasing order).
    postings
}

/// AP-biased synthesizer: emulates the regime D2 targets — a locality
/// reorder on cluster postings so IDs within a posting are *near*
/// arithmetic progression (constant stride plus small jitter).
///
/// This mimics: Hilbert-reordered dataset + per-cluster sort, where
/// adjacent IDs in a posting come from physically-adjacent slots.
///
/// `jitter` is the half-width of the added integer noise on each step.
pub fn synth_postings_ap_biased(
    n_total: usize,
    num_clusters: usize,
    jitter: i64,
    seed: u64,
) -> Vec<Vec<u32>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut postings: Vec<Vec<u32>> = vec![Vec::new(); num_clusters];
    // Assign each cluster its own sub-range of the global ID space and
    // march with roughly constant stride through it. After reorder, all
    // IDs for a cluster live in a contiguous slab of the dataset.
    let slab = n_total / num_clusters;
    for c in 0..num_clusters {
        let base = c * slab;
        let mut cur = base as i64;
        for _ in 0..slab {
            if cur < 0 || cur >= n_total as i64 {
                break;
            }
            postings[c].push(cur as u32);
            let j = if jitter > 0 {
                rng.gen_range(-jitter..=jitter)
            } else {
                0
            };
            cur += 1 + j;
        }
    }
    postings
}

/// Sparse-ID AP-biased synthesizer: emulates a 32-bit global ID namespace
/// where posting IDs come from a specific tenant/shard stride.
/// Each posting is a near-AP sequence with a *large* mean stride, so
/// single deltas no longer fit in 7 bits — this is the regime where
/// delta-of-delta beats single-delta varint.
///
///  * `stride`: mean per-ID stride in the global namespace.
///  * `jitter`: half-width of integer noise added per step.
pub fn synth_postings_sparse_ap(
    n_per_posting: usize,
    num_clusters: usize,
    stride: i64,
    jitter: i64,
    seed: u64,
) -> Vec<Vec<u32>> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut postings: Vec<Vec<u32>> = vec![Vec::with_capacity(n_per_posting); num_clusters];
    for c in 0..num_clusters {
        let base = (c as u64).wrapping_mul(0x9E3779B97F4A7C15) as u32;
        let mut cur = base as i64;
        for _ in 0..n_per_posting {
            let v = cur as u32;
            postings[c].push(v);
            let j = if jitter > 0 {
                rng.gen_range(-jitter..=jitter)
            } else {
                0
            };
            cur = cur.wrapping_add(stride + j);
        }
        postings[c].sort_unstable();
        postings[c].dedup();
    }
    postings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<C: PostingCodec + ?Sized>(c: &C, ids: &[u32]) {
        let bytes = c.encode(ids);
        let back = c.decode(&bytes);
        assert_eq!(back, ids, "roundtrip failed for {}", c.name());
    }

    #[test]
    fn empty() {
        for c in codecs() {
            roundtrip(&*c, &[]);
        }
    }

    #[test]
    fn single() {
        for c in codecs() {
            roundtrip(&*c, &[42]);
            roundtrip(&*c, &[u32::MAX / 2]);
        }
    }

    #[test]
    fn increasing_small() {
        let ids: Vec<u32> = (0..100).collect();
        for c in codecs() {
            roundtrip(&*c, &ids);
        }
    }

    #[test]
    fn arithmetic_progression() {
        // Perfect DoD=0: VarintDelta emits zz(7) per step (1 byte each),
        // D2Varint emits zz(0) per step (1 byte each) — tied at 1 B/ID.
        // D2 wins when the common delta is LARGE enough that VarintDelta
        // needs a 2-byte varint but D2 still emits 0x00.
        let ids: Vec<u32> = (0..1000).map(|i| 1000 + i * 1000).collect();
        for c in codecs() {
            roundtrip(&*c, &ids);
        }
        let raw = RawU32.encode(&ids).len();
        let vd = VarintDelta.encode(&ids).len();
        let d2 = D2Varint.encode(&ids).len();
        assert!(d2 < vd, "D2 ({}) should beat varint-delta ({}) when delta>=128", d2, vd);
        assert!(vd < raw, "varint-delta ({}) should beat raw ({})", vd, raw);
    }

    #[test]
    fn synth_matches() {
        let postings = synth_postings(10_000, 64, 1);
        for p in postings.iter().take(5) {
            for c in codecs() {
                roundtrip(&*c, p);
            }
        }
    }

    fn codecs() -> Vec<Box<dyn PostingCodec>> {
        vec![Box::new(RawU32), Box::new(VarintDelta), Box::new(D2Varint)]
    }
}
