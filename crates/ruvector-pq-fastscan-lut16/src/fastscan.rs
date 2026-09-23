//! FastScan LUT16 backends: 4-bit codes + u8 LUT + byte-shuffle scan.

use crate::codebook::{build_lut16_u8, encode_one, top_k, train_codebook, Lut16U8, SubCodebook};
use crate::{PqAdc, PqError, BLOCK, KS16};

/// FastScan LUT16 backend without block transposition.
///
/// Each vector stores `m/2` bytes (two 4-bit codes per byte). The scan loop
/// is a per-vector sum-over-m using a u8 LUT indexed by the nibble. Same
/// access pattern as the baseline, half the code footprint.
pub struct Lut16ScalarFastscan {
    books: Vec<SubCodebook>,
    packed: Vec<u8>,
    n: usize,
    m: usize,
}

impl Lut16ScalarFastscan {
    pub fn build(
        data: &[f32],
        n: usize,
        d: usize,
        m: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self, PqError> {
        if m % 2 != 0 {
            return Err(PqError::BadShape { d, m });
        }
        let books = train_codebook(data, n, d, m, KS16, iters, seed)?;
        let mut packed = vec![0u8; n * (m / 2)];
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            let codes = encode_one(x, &books);
            for sq in 0..m {
                let byte = i * (m / 2) + sq / 2;
                let nib = codes[sq] & 0x0f;
                if sq % 2 == 0 {
                    packed[byte] |= nib;
                } else {
                    packed[byte] |= nib << 4;
                }
            }
        }
        Ok(Self { books, packed, n, m })
    }
}

impl PqAdc for Lut16ScalarFastscan {
    fn len(&self) -> usize { self.n }
    fn code_bytes(&self) -> usize { self.packed.len() }
    fn m(&self) -> usize { self.m }
    fn ks(&self) -> usize { KS16 }
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let lut = build_lut16_u8(&self.books, query);
        let mhalf = self.m / 2;
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let row = &self.packed[i * mhalf..(i + 1) * mhalf];
            let mut acc: u16 = 0;
            for j in 0..mhalf {
                let byte = row[j];
                let lo = (byte & 0x0f) as usize;
                let hi = (byte >> 4) as usize;
                acc += lut.lut[(2 * j) * KS16 + lo] as u16;
                acc += lut.lut[(2 * j + 1) * KS16 + hi] as u16;
            }
            dists.push((i as u32, acc as f32 * lut.scale));
        }
        top_k(dists, k)
    }
}

/// FastScan LUT16 with block-transposed layout.
///
/// Vectors are grouped into blocks of BLOCK=32. For each sub-quantizer we
/// store all 32 codes as 16 bytes: low nibble = lane t in 0..16, high nibble
/// = lane t+16 in 16..32. This is the Faiss "fastscan block4" / SCANN LUT16
/// layout: a single 16-lane byte-shuffle (`VPSHUFB` on x86, `TBL` on aarch64)
/// resolves 16 lookups at once.
pub struct Lut16BlockedFastscan {
    books: Vec<SubCodebook>,
    blocks: Vec<Block>,
    n: usize,
    m: usize,
}

struct Block {
    sub: Vec<[u8; 16]>, // len == m
}

impl Lut16BlockedFastscan {
    pub fn build(
        data: &[f32],
        n: usize,
        d: usize,
        m: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self, PqError> {
        let books = train_codebook(data, n, d, m, KS16, iters, seed)?;
        let nblocks = n.div_ceil(BLOCK);
        let mut blocks = Vec::with_capacity(nblocks);
        for b in 0..nblocks {
            let mut sub = Vec::with_capacity(m);
            for _ in 0..m {
                sub.push([0u8; 16]);
            }
            for t in 0..BLOCK {
                let i = b * BLOCK + t;
                if i >= n {
                    break;
                }
                let x = &data[i * d..(i + 1) * d];
                let codes = encode_one(x, &books);
                for sq in 0..m {
                    let nib = codes[sq] & 0x0f;
                    if t < 16 {
                        sub[sq][t] |= nib;
                    } else {
                        sub[sq][t - 16] |= nib << 4;
                    }
                }
            }
            blocks.push(Block { sub });
        }
        Ok(Self { books, blocks, n, m })
    }

    fn scan_block(&self, lut: &Lut16U8, block: &Block, acc: &mut [u16; BLOCK]) {
        for sq in 0..self.m {
            let table = &lut.lut[sq * KS16..sq * KS16 + KS16];
            let col = &block.sub[sq];
            for t in 0..16 {
                let b = col[t];
                let lo = (b & 0x0f) as usize;
                let hi = (b >> 4) as usize;
                acc[t] = acc[t].wrapping_add(table[lo] as u16);
                acc[t + 16] = acc[t + 16].wrapping_add(table[hi] as u16);
            }
        }
    }
}

impl PqAdc for Lut16BlockedFastscan {
    fn len(&self) -> usize { self.n }
    fn code_bytes(&self) -> usize { self.blocks.len() * self.m * 16 }
    fn m(&self) -> usize { self.m }
    fn ks(&self) -> usize { KS16 }
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let lut = build_lut16_u8(&self.books, query);
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(self.n);
        for (b, block) in self.blocks.iter().enumerate() {
            let mut acc = [0u16; BLOCK];
            self.scan_block(&lut, block, &mut acc);
            for t in 0..BLOCK {
                let i = b * BLOCK + t;
                if i >= self.n {
                    break;
                }
                dists.push((i as u32, acc[t] as f32 * lut.scale));
            }
        }
        top_k(dists, k)
    }
}
