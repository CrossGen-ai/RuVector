//! Bit-packed code layout. Codes 2..=8 bits each; up to 8 bits per field
//! means every read/write spans at most 2 bytes.

#[inline]
pub fn write_bits(buf: &mut [u8], bit_off: u32, nbits: u8, val: u32) {
    debug_assert!(nbits <= 8);
    debug_assert!(val < (1u32 << nbits));
    let byte = (bit_off / 8) as usize;
    let shift = (bit_off % 8) as u32;
    let v = val << shift; // up to 16 bits worth
    buf[byte] |= (v & 0xFF) as u8;
    if shift + nbits as u32 > 8 {
        buf[byte + 1] |= ((v >> 8) & 0xFF) as u8;
    }
}

#[inline]
pub fn read_bits(buf: &[u8], bit_off: u32, nbits: u8) -> u32 {
    debug_assert!(nbits <= 8);
    let byte = (bit_off / 8) as usize;
    let shift = (bit_off % 8) as u32;
    let mut v = buf[byte] as u32;
    if shift + nbits as u32 > 8 {
        v |= (buf[byte + 1] as u32) << 8;
    }
    (v >> shift) & ((1u32 << nbits) - 1)
}
