//! PQ-then-rerank baseline. Score all codes with ADT, keep top `rerank_k`,
//! then rescore those with full-precision L2.

use crate::{metric::l2_sq, pq::{PqCodes, ProductQuantizer}, AnnIndex};

pub struct PqRerankIndex {
    pub d: usize,
    pub data: Vec<f32>,
    pub pq: ProductQuantizer,
    pub codes: PqCodes,
    pub rerank_k: usize,
}

impl PqRerankIndex {
    pub fn build(d: usize, data: Vec<f32>, m: usize, k: usize, iters: usize, rerank_k: usize, seed: u64) -> Self {
        let pq = ProductQuantizer::train(&data, d, m, k, iters, seed).expect("train");
        let codes = pq.encode_all(&data).expect("encode");
        Self { d, data, pq, codes, rerank_k }
    }
    fn n(&self) -> usize { self.data.len() / self.d }
}

impl AnnIndex for PqRerankIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let lut = self.pq.build_adt(query);
        let n = self.n();
        let mut est: Vec<(u32, f32)> = (0..n)
            .map(|i| (i as u32, self.pq.adc_distance(&lut, self.codes.code(i))))
            .collect();
        let take = self.rerank_k.min(n);
        // Partial sort.
        est.select_nth_unstable_by(take.saturating_sub(1).max(0), |a, b| a.1.partial_cmp(&b.1).unwrap());
        est.truncate(take);
        // Rerank with full precision.
        let mut rer: Vec<(u32, f32)> = est.into_iter()
            .map(|(id, _)| (id, l2_sq(&self.data[id as usize * self.d..(id as usize + 1) * self.d], query)))
            .collect();
        rer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        rer.truncate(k);
        rer
    }
    fn len(&self) -> usize { self.n() }
}
