//! Delta + Frame-of-Reference bitpacked block codec (PFOR-lite).
//!
//! Layout:
//!   [varint n] [u32_le first_id] [ block ... ]
//! Each block:
//!   [u8 block_len (1..=128)] [u8 bit_width (0..=32)]
//!   [ceil(block_len * bit_width / 8) bytes of little-endian bitpacked gaps]
//!
//! Blocks hold *gaps* between successive ids, not the ids themselves. The
//! first id of the list is stored raw to anchor the running sum.

use crate::{assert_sorted, PostingCodec};

const BLOCK: usize = 128;

#[inline]
fn bit_width_for_max(max: u32) -> u8 {
    if max == 0 {
        0
    } else {
        (32 - max.leading_zeros()) as u8
    }
}

/// Write `b` bits of `value` at `bit_pos` into `buf` (little-endian within
/// each byte: bit 0 of value lands at bit `bit_pos % 8` of
/// `buf[bit_pos / 8]`). Assumes `buf` is sized large enough.
#[inline]
fn put_bits(buf: &mut [u8], bit_pos: usize, b: u8, value: u32) {
    if b == 0 {
        return;
    }
    let mut byte_index = bit_pos / 8;
    let bit_offset = bit_pos % 8;
    // Shift value up by bit_offset into a wider accumulator.
    let v = (value as u64) << bit_offset;
    let total_bits = b as usize + bit_offset;
    let bytes_to_touch = (total_bits + 7) / 8;
    for i in 0..bytes_to_touch {
        buf[byte_index] |= ((v >> (i * 8)) & 0xFF) as u8;
        byte_index += 1;
    }
}

#[inline]
fn get_bits(buf: &[u8], bit_pos: usize, b: u8) -> u32 {
    if b == 0 {
        return 0;
    }
    let byte_index = bit_pos / 8;
    let bit_offset = bit_pos % 8;
    let bytes_needed = (b as usize + bit_offset + 7) / 8;
    let mut acc: u64 = 0;
    for i in 0..bytes_needed {
        acc |= (buf[byte_index + i] as u64) << (i * 8);
    }
    let mask: u64 = if b == 32 {
        0xFFFF_FFFF
    } else {
        (1u64 << b) - 1
    };
    ((acc >> bit_offset) & mask) as u32
}

#[inline]
fn write_varint(buf: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        buf.push(((v & 0x7F) as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

#[inline]
fn read_varint(bytes: &[u8], pos: &mut usize) -> u32 {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let b = bytes[*pos];
        *pos += 1;
        result |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            return result;
        }
        shift += 7;
    }
}

pub struct PforCodec;

impl PostingCodec for PforCodec {
    fn encode(ids: &[u32]) -> Vec<u8> {
        assert_sorted(ids);
        let mut out = Vec::with_capacity(ids.len() * 2 + 16);
        write_varint(&mut out, ids.len() as u32);
        if ids.is_empty() {
            return out;
        }
        out.extend_from_slice(&ids[0].to_le_bytes());

        let mut prev = ids[0];
        let gaps_total = ids.len() - 1;
        let mut i = 0usize;
        while i < gaps_total {
            let block_len = (gaps_total - i).min(BLOCK);
            let gap_slice_end = i + block_len;
            // Compute max gap in this block.
            let mut max_gap: u32 = 0;
            let mut p = prev;
            for &x in &ids[1 + i..1 + gap_slice_end] {
                let g = x - p;
                if g > max_gap {
                    max_gap = g;
                }
                p = x;
            }
            let b = bit_width_for_max(max_gap);
            out.push(block_len as u8);
            out.push(b);
            let payload_bytes = ((block_len * b as usize) + 7) / 8;
            let start = out.len();
            out.resize(start + payload_bytes, 0);
            let mut bit_pos = 0usize;
            let mut p2 = prev;
            for &x in &ids[1 + i..1 + gap_slice_end] {
                let g = x - p2;
                put_bits(&mut out[start..start + payload_bytes], bit_pos, b, g);
                bit_pos += b as usize;
                p2 = x;
            }
            prev = p2;
            i = gap_slice_end;
        }

        out
    }

    fn decode(bytes: &[u8]) -> Vec<u32> {
        let mut pos = 0usize;
        let n = read_varint(bytes, &mut pos) as usize;
        let mut out = Vec::with_capacity(n);
        if n == 0 {
            return out;
        }
        let first = u32::from_le_bytes([
            bytes[pos],
            bytes[pos + 1],
            bytes[pos + 2],
            bytes[pos + 3],
        ]);
        pos += 4;
        out.push(first);
        let mut prev = first;
        let mut remaining = n - 1;
        while remaining > 0 {
            let block_len = bytes[pos] as usize;
            pos += 1;
            let b = bytes[pos];
            pos += 1;
            let payload_bytes = ((block_len * b as usize) + 7) / 8;
            let payload = &bytes[pos..pos + payload_bytes];
            pos += payload_bytes;
            let mut bit_pos = 0usize;
            for _ in 0..block_len {
                let g = get_bits(payload, bit_pos, b);
                bit_pos += b as usize;
                prev = prev.wrapping_add(g);
                out.push(prev);
            }
            remaining -= block_len;
        }
        out
    }

    fn name() -> &'static str {
        "delta+pfor-128"
    }
}

#[cfg(test)]
mod bit_tests {
    use super::*;

    #[test]
    fn put_get_bits_roundtrip() {
        let mut buf = vec![0u8; 64];
        let widths = [1u8, 3, 7, 8, 12, 15, 17, 23, 31, 32];
        for &b in &widths {
            buf.iter_mut().for_each(|x| *x = 0);
            let mask: u64 = if b == 32 { u32::MAX as u64 } else { (1u64 << b) - 1 };
            let mut bit_pos = 0usize;
            let mut vals = Vec::new();
            for i in 0..7 {
                let v = ((i as u64).wrapping_mul(0x9E37_79B1) & mask) as u32;
                put_bits(&mut buf, bit_pos, b, v);
                bit_pos += b as usize;
                vals.push(v);
            }
            let mut bit_pos = 0usize;
            for v in vals {
                let got = get_bits(&buf, bit_pos, b);
                assert_eq!(got, v, "b={b}");
                bit_pos += b as usize;
            }
        }
    }
}
