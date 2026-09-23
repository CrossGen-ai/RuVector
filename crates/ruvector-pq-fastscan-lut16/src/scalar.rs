//! Baseline scalar PQ-ADC (Jegou 2011): 8-bit codes, f32 LUT.

use crate::codebook::{encode_one, sq_l2, top_k, train_codebook, SubCodebook};
use crate::{PqAdc, PqError};

/// Baseline PQ-ADC. `ks = 256`, one byte per sub-quantizer per vector, f32 LUT.
pub struct ScalarPqAdc {
    books: Vec<SubCodebook>,
    codes: Vec<u8>, // row-major [n * m]
    n: usize,
    m: usize,
    ks: usize,
}

impl ScalarPqAdc {
    pub fn build(
        data: &[f32],
        n: usize,
        d: usize,
        m: usize,
        ks: usize,
        iters: usize,
        seed: u64,
    ) -> Result<Self, PqError> {
        let books = train_codebook(data, n, d, m, ks, iters, seed)?;
        for b in &books {
            if b.ks() != ks {
                return Err(PqError::BadKs {
                    expected: ks,
                    got: b.ks(),
                });
            }
        }
        let mut codes = vec![0u8; n * m];
        for i in 0..n {
            let x = &data[i * d..(i + 1) * d];
            let c = encode_one(x, &books);
            codes[i * m..(i + 1) * m].copy_from_slice(&c);
        }
        Ok(Self { books, codes, n, m, ks })
    }

    fn build_lut_f32(&self, query: &[f32]) -> Vec<f32> {
        let mut lut = vec![0f32; self.m * self.ks];
        let mut off = 0usize;
        for sq in 0..self.m {
            let ds = self.books[sq].ds;
            let qsub = &query[off..off + ds];
            for c in 0..self.ks {
                lut[sq * self.ks + c] = sq_l2(qsub, &self.books[sq].centroids[c]);
            }
            off += ds;
        }
        lut
    }
}

impl PqAdc for ScalarPqAdc {
    fn len(&self) -> usize { self.n }
    fn code_bytes(&self) -> usize { self.codes.len() }
    fn m(&self) -> usize { self.m }
    fn ks(&self) -> usize { self.ks }
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let lut = self.build_lut_f32(query);
        let mut dists: Vec<(u32, f32)> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let row = &self.codes[i * self.m..(i + 1) * self.m];
            let mut d = 0.0f32;
            for sq in 0..self.m {
                d += lut[sq * self.ks + row[sq] as usize];
            }
            dists.push((i as u32, d));
        }
        top_k(dists, k)
    }
}
