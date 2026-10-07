//! Compressed IVF posting-list codecs.
//!
//! Three swappable codecs encode/decode a sorted (ascending) `&[u32]` posting
//! list of document ids:
//!
//! * `RawCodec`    — baseline: `4 * N` bytes, no transform.
//! * `VarintCodec` — delta + byte-aligned varint; small, fully portable.
//! * `PforCodec`   — delta + fixed-width bitpacked blocks (Patched
//!   Frame-of-Reference, 128-element blocks, byte-aligned bit-width header).
//!
//! Decode produces the original `Vec<u32>` so an IVF scan can reuse any
//! existing distance kernel.

#![forbid(unsafe_code)]

pub mod pfor;
pub mod raw;
pub mod scan;
pub mod varint;

/// A posting-list codec: encodes a sorted list of document ids into a
/// byte buffer and decodes it back.
pub trait PostingCodec {
    /// Encode a sorted ascending slice of ids. Panics if `ids` is not sorted.
    fn encode(ids: &[u32]) -> Vec<u8>;

    /// Decode a buffer previously produced by `encode` back into ids.
    fn decode(bytes: &[u8]) -> Vec<u32>;

    /// Codec name for reports.
    fn name() -> &'static str;
}

/// Guard — debug-only; `encode` assumes sorted input.
#[inline]
pub(crate) fn assert_sorted(ids: &[u32]) {
    debug_assert!(ids.windows(2).all(|w| w[0] <= w[1]), "ids must be sorted");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<u32> {
        let mut v = Vec::with_capacity(1000);
        let mut cur: u32 = 7;
        for i in 0..1000 {
            cur = cur.wrapping_add(1 + (i as u32 % 37));
            v.push(cur);
        }
        v.sort();
        v
    }

    #[test]
    fn roundtrip_raw() {
        let ids = sample();
        let b = raw::RawCodec::encode(&ids);
        assert_eq!(raw::RawCodec::decode(&b), ids);
    }

    #[test]
    fn roundtrip_varint() {
        let ids = sample();
        let b = varint::VarintCodec::encode(&ids);
        assert_eq!(varint::VarintCodec::decode(&b), ids);
    }

    #[test]
    fn roundtrip_pfor() {
        let ids = sample();
        let b = pfor::PforCodec::encode(&ids);
        assert_eq!(pfor::PforCodec::decode(&b), ids);
    }

    #[test]
    fn roundtrip_pfor_edges() {
        // sizes that straddle block boundaries
        for n in [0, 1, 127, 128, 129, 255, 256, 257, 1000] {
            let ids: Vec<u32> = (0..n as u32).map(|i| i * 3 + 100).collect();
            let b = pfor::PforCodec::encode(&ids);
            let d = pfor::PforCodec::decode(&b);
            assert_eq!(d, ids, "mismatch at n={n}");
        }
    }

    #[test]
    fn roundtrip_varint_empty() {
        let b = varint::VarintCodec::encode(&[]);
        assert!(varint::VarintCodec::decode(&b).is_empty());
    }
}
