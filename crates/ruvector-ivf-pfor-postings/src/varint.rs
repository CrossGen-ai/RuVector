//! Delta + unsigned LEB128 varint.
//!
//! `ids` are sorted ascending: emit `ids[0]`, then successive gaps
//! `g_i = ids[i] - ids[i-1]`, each as LEB128. Decode reverses the running sum.

use crate::{assert_sorted, PostingCodec};

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

pub struct VarintCodec;

impl PostingCodec for VarintCodec {
    fn encode(ids: &[u32]) -> Vec<u8> {
        assert_sorted(ids);
        // Length prefix as varint, then first id, then gaps.
        let mut out = Vec::with_capacity(ids.len() + 8);
        write_varint(&mut out, ids.len() as u32);
        if ids.is_empty() {
            return out;
        }
        write_varint(&mut out, ids[0]);
        let mut prev = ids[0];
        for &x in &ids[1..] {
            write_varint(&mut out, x - prev);
            prev = x;
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
        let first = read_varint(bytes, &mut pos);
        out.push(first);
        let mut prev = first;
        for _ in 1..n {
            let g = read_varint(bytes, &mut pos);
            prev = prev.wrapping_add(g);
            out.push(prev);
        }
        out
    }

    fn name() -> &'static str {
        "delta+varint"
    }
}
