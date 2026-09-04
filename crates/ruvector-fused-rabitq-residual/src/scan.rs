//! Quantizer-agnostic linear scan index.

use crate::quantizer::Quantizer;

pub struct QuantizedIndex<Q: Quantizer> {
    pub q: Q,
    pub codes: Vec<u8>,
    pub n: usize,
    pub code_len: usize,
}

impl<Q: Quantizer> QuantizedIndex<Q> {
    pub fn build(q: Q, vectors: &[Vec<f32>]) -> Self {
        let code_len = q.code_bytes();
        let n = vectors.len();
        let mut codes = Vec::with_capacity(n * code_len);
        for v in vectors {
            codes.extend(q.encode(v));
        }
        Self {
            q,
            codes,
            n,
            code_len,
        }
    }

    pub fn topk(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let ctx = self.q.prepare_query(query);
        let mut scored: Vec<(usize, f32)> = (0..self.n)
            .map(|i| {
                let code = &self.codes[i * self.code_len..(i + 1) * self.code_len];
                (i, self.q.distance(code, &ctx))
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        scored
    }
}
