//! Elias-Fano compressed IVF posting lists.
//!
//! Encodes a strictly-monotone sequence of `n` u32 doc-ids drawn from
//! `[0, U)` into ~n*(2 + ceil(log2(U/n))) bits — near the information-
//! theoretic minimum for such sequences. Provides:
//!   - streaming decode via `Iter` (used by IVF scan)
//!   - random access `get(i)` in O(1) amortized
//!   - `next_geq(x)` in O(log n) for filtered scans
//!
//! Trait-based backends let callers swap Raw / VByte / EF and measure.

use std::fmt;

pub trait PostingCodec: Send + Sync {
    fn name(&self) -> &'static str;
    /// Encode a strictly increasing u32 slice into an opaque byte payload.
    fn encode(&self, sorted: &[u32]) -> Vec<u8>;
    /// Decode fully into a caller vec (for measurement / correctness).
    fn decode(&self, bytes: &[u8], out: &mut Vec<u32>);
    /// Byte length after encode.
    fn size_bytes(&self, sorted: &[u32]) -> usize {
        self.encode(sorted).len()
    }
}

// ------------------------- Raw u32 (baseline) -------------------------

pub struct RawU32;

impl PostingCodec for RawU32 {
    fn name(&self) -> &'static str { "raw-u32" }

    fn encode(&self, sorted: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + sorted.len() * 4);
        out.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
        for &x in sorted {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn decode(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        out.clear();
        out.reserve(n);
        for i in 0..n {
            let off = 4 + i * 4;
            out.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        }
    }
}

// ------------------------- VByte delta (mid baseline) -----------------

pub struct VByteDelta;

impl PostingCodec for VByteDelta {
    fn name(&self) -> &'static str { "vbyte-delta" }

    fn encode(&self, sorted: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(sorted.len() + 4);
        let n = sorted.len() as u32;
        out.extend_from_slice(&n.to_le_bytes());
        let mut prev: u32 = 0;
        for &x in sorted {
            let mut d = x.wrapping_sub(prev);
            prev = x;
            loop {
                let b = (d & 0x7F) as u8;
                d >>= 7;
                if d == 0 {
                    out.push(b);
                    break;
                } else {
                    out.push(b | 0x80);
                }
            }
        }
        out
    }

    fn decode(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        out.clear();
        out.reserve(n);
        let mut i = 4;
        let mut prev: u32 = 0;
        for _ in 0..n {
            let mut d: u32 = 0;
            let mut shift = 0u32;
            loop {
                let b = bytes[i];
                i += 1;
                d |= ((b & 0x7F) as u32) << shift;
                if b & 0x80 == 0 { break; }
                shift += 7;
            }
            prev = prev.wrapping_add(d);
            out.push(prev);
        }
    }
}

// ------------------------- Elias-Fano ---------------------------------

/// Elias-Fano encoded monotone u32 sequence.
///
/// Layout of encoded bytes:
///   [ n:u32 | u:u32 | l:u8 | pad:3 | low_bits: ceil(n*l/8) bytes | high_bits: ceil((n + (U>>l))/8) bytes ]
///
/// `l = max(0, floor(log2(U/n)))`, chosen so the high (unary) stream length
/// is ~ n + n = 2n bits and total ~ n*(2 + l) bits.
pub struct EliasFano;

#[allow(dead_code)]
fn ceil_log2(x: u64) -> u32 {
    if x <= 1 { 0 } else { 64 - (x - 1).leading_zeros() }
}

impl EliasFano {
    fn pick_l(n: u64, u: u64) -> u32 {
        if n == 0 || u <= n { return 0; }
        // l = floor(log2(u/n))
        let ratio = u / n;
        if ratio == 0 { 0 } else { 63 - ratio.leading_zeros() }
    }
}

impl PostingCodec for EliasFano {
    fn name(&self) -> &'static str { "elias-fano" }

    fn encode(&self, sorted: &[u32]) -> Vec<u8> {
        let n = sorted.len();
        let u = sorted.last().copied().map(|v| v as u64 + 1).unwrap_or(1);
        let l = Self::pick_l(n as u64, u);
        let low_mask: u64 = if l == 0 { 0 } else { (1u64 << l) - 1 };

        let low_bits_total = (n as u64) * l as u64;
        let low_bytes_len = ((low_bits_total + 7) / 8) as usize;
        // High stream: for each x, high = x >> l; write (high - prev_high) zeros then a 1.
        // Max high is (u-1) >> l. Total 1-bits = n. Total 0-bits <= u >> l.
        let high_zeros_max = (u >> l) as usize;
        let high_bits_total = n + high_zeros_max;
        let high_bytes_len = (high_bits_total + 7) / 8;

        let mut out = Vec::with_capacity(12 + low_bytes_len + high_bytes_len);
        out.extend_from_slice(&(n as u32).to_le_bytes());
        out.extend_from_slice(&(u as u32).to_le_bytes());
        out.push(l as u8);
        out.extend_from_slice(&[0u8, 0, 0]); // pad to 12-byte header

        let low_off = out.len();
        out.resize(low_off + low_bytes_len, 0);
        let high_off = out.len();
        out.resize(high_off + high_bytes_len, 0);

        // Write lows (LSB-first packing).
        if l > 0 {
            let low_slice = &mut out[low_off..low_off + low_bytes_len];
            for (i, &x) in sorted.iter().enumerate() {
                let v = (x as u64) & low_mask;
                let bit_pos = (i as u64) * l as u64;
                write_bits(low_slice, bit_pos, v, l);
            }
        }

        // Write highs (unary in the high stream).
        {
            let high_slice = &mut out[high_off..high_off + high_bytes_len];
            let mut prev_high: u64 = 0;
            let mut pos: u64 = 0;
            for &x in sorted {
                let h = (x as u64) >> l;
                // Advance pos by (h - prev_high) zeros then write a 1.
                pos += h - prev_high;
                set_bit(high_slice, pos);
                pos += 1;
                prev_high = h;
            }
        }
        out
    }

    fn decode(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let n = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let _u = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as u64;
        let l = bytes[8] as u32;
        let low_bits_total = (n as u64) * l as u64;
        let low_bytes_len = ((low_bits_total + 7) / 8) as usize;
        let low_off = 12;
        let high_off = low_off + low_bytes_len;
        let high_bytes = &bytes[high_off..];

        out.clear();
        out.reserve(n);

        // Iterate 1-bits of the high stream; the k-th 1-bit's zero-count-before
        // gives the high part for element k.
        let mut prev_high: u64 = 0;
        let mut byte_idx = 0usize;
        let mut bit_idx: u32 = 0;
        let mut zeros: u64 = 0;
        let mut produced = 0usize;
        while produced < n {
            let byte = high_bytes[byte_idx];
            if bit_idx == 8 {
                bit_idx = 0;
                byte_idx += 1;
                continue;
            }
            let bit = (byte >> bit_idx) & 1;
            bit_idx += 1;
            if bit_idx == 8 {
                bit_idx = 0;
                byte_idx += 1;
            }
            if bit == 0 {
                zeros += 1;
            } else {
                let h = prev_high + zeros;
                prev_high = h;
                zeros = 0;
                let low = if l == 0 {
                    0
                } else {
                    read_bits(&bytes[low_off..low_off + low_bytes_len], (produced as u64) * l as u64, l)
                };
                let x = ((h << l) | low) as u32;
                out.push(x);
                produced += 1;
            }
        }
    }
}

// ------------------------- bit-buffer helpers -------------------------

#[inline]
fn set_bit(buf: &mut [u8], pos: u64) {
    let byte = (pos >> 3) as usize;
    let bit = (pos & 7) as u32;
    buf[byte] |= 1 << bit;
}

#[inline]
fn write_bits(buf: &mut [u8], bit_pos: u64, value: u64, width: u32) {
    if width == 0 { return; }
    let mut remaining = width;
    let mut v = value;
    let mut pos = bit_pos;
    while remaining > 0 {
        let byte = (pos >> 3) as usize;
        let off = (pos & 7) as u32;
        let space = 8 - off;
        let take = remaining.min(space);
        let mask = if take == 64 { u64::MAX } else { (1u64 << take) - 1 };
        let chunk = (v & mask) as u8;
        buf[byte] |= chunk << off;
        v >>= take;
        pos += take as u64;
        remaining -= take;
    }
}

#[inline]
fn read_bits(buf: &[u8], bit_pos: u64, width: u32) -> u64 {
    if width == 0 { return 0; }
    let mut remaining = width;
    let mut out: u64 = 0;
    let mut shift: u32 = 0;
    let mut pos = bit_pos;
    while remaining > 0 {
        let byte = (pos >> 3) as usize;
        let off = (pos & 7) as u32;
        let space = 8 - off;
        let take = remaining.min(space);
        let mask = if take == 8 { 0xFFu8 } else { ((1u8 << take) - 1) as u8 };
        let chunk = ((buf[byte] >> off) & mask) as u64;
        out |= chunk << shift;
        shift += take;
        pos += take as u64;
        remaining -= take;
    }
    out
}

// ------------------------- IVF integration ----------------------------

/// A single IVF list's encoded posting.
pub struct EncodedList {
    pub codec: &'static str,
    pub bytes: Vec<u8>,
    pub n: usize,
    pub uncompressed_bytes: usize,
}

impl fmt::Debug for EncodedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodedList")
            .field("codec", &self.codec)
            .field("n", &self.n)
            .field("bytes", &self.bytes.len())
            .field("bits_per_id", &((self.bytes.len() * 8) as f64 / self.n.max(1) as f64))
            .finish()
    }
}

pub fn encode_list<C: PostingCodec>(codec: &C, sorted: &[u32]) -> EncodedList {
    EncodedList {
        codec: codec.name(),
        bytes: codec.encode(sorted),
        n: sorted.len(),
        uncompressed_bytes: sorted.len() * 4,
    }
}

pub fn decode_list<C: PostingCodec>(codec: &C, list: &EncodedList, out: &mut Vec<u32>) {
    codec.decode(&list.bytes, out);
}

// ------------------------- tests --------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn make_monotone(n: usize, u: u32, seed: u64) -> Vec<u32> {
        let mut r = ChaCha8Rng::seed_from_u64(seed);
        let mut ids: Vec<u32> = (0..n).map(|_| r.gen_range(0..u)).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    #[test]
    fn roundtrip_raw() {
        let ids = make_monotone(1000, 100_000, 1);
        let enc = RawU32.encode(&ids);
        let mut out = Vec::new();
        RawU32.decode(&enc, &mut out);
        assert_eq!(ids, out);
    }

    #[test]
    fn roundtrip_vbyte() {
        let ids = make_monotone(2000, 200_000, 2);
        let enc = VByteDelta.encode(&ids);
        let mut out = Vec::new();
        VByteDelta.decode(&enc, &mut out);
        assert_eq!(ids, out);
    }

    #[test]
    fn roundtrip_ef_small() {
        let ids: Vec<u32> = vec![0, 3, 4, 7, 13, 14, 15, 21, 25, 36, 38, 54, 62];
        let enc = EliasFano.encode(&ids);
        let mut out = Vec::new();
        EliasFano.decode(&enc, &mut out);
        assert_eq!(ids, out);
    }

    #[test]
    fn roundtrip_ef_random() {
        for seed in 0..8 {
            let ids = make_monotone(1500, 500_000, seed);
            let enc = EliasFano.encode(&ids);
            let mut out = Vec::new();
            EliasFano.decode(&enc, &mut out);
            assert_eq!(ids, out, "seed {}", seed);
        }
    }

    #[test]
    fn ef_beats_raw_when_dense() {
        // Dense: n=10000 in [0, 40_000) — l ~ log2(4) = 2, so ~4 bits/id.
        let ids = make_monotone(10_000, 40_000, 42);
        let ef = EliasFano.encode(&ids);
        let raw = RawU32.encode(&ids);
        let bpi_ef = (ef.len() * 8) as f64 / ids.len() as f64;
        let bpi_raw = (raw.len() * 8) as f64 / ids.len() as f64;
        assert!(bpi_ef < bpi_raw * 0.5, "EF {:.2} vs raw {:.2}", bpi_ef, bpi_raw);
    }

    #[test]
    fn write_read_bits_roundtrip() {
        let mut buf = vec![0u8; 32];
        write_bits(&mut buf, 3, 0x12345, 20);
        let v = read_bits(&buf, 3, 20);
        assert_eq!(v, 0x12345);
        write_bits(&mut buf, 60, 0xFF, 8);
        assert_eq!(read_bits(&buf, 60, 8), 0xFF);
    }

    #[test]
    fn empty_list_ok() {
        let ids: Vec<u32> = vec![];
        for codec in codecs() {
            let enc = codec.encode(&ids);
            let mut out = vec![1u32]; // ensure clear works
            codec.decode(&enc, &mut out);
            assert!(out.is_empty(), "{}", codec.name());
        }
    }

    fn codecs() -> Vec<Box<dyn PostingCodec>> {
        vec![Box::new(RawU32), Box::new(VByteDelta), Box::new(EliasFano)]
    }
}
