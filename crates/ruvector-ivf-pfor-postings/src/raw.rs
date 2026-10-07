//! Baseline: little-endian u32 array.

use crate::{assert_sorted, PostingCodec};

pub struct RawCodec;

impl PostingCodec for RawCodec {
    fn encode(ids: &[u32]) -> Vec<u8> {
        assert_sorted(ids);
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &x in ids {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn decode(bytes: &[u8]) -> Vec<u32> {
        let n = bytes.len() / 4;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let b = &bytes[i * 4..i * 4 + 4];
            out.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
        out
    }

    fn name() -> &'static str {
        "raw-u32"
    }
}
