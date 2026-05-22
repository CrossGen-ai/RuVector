//! The SymphonyQG index: graph + quantized codes + full-precision vectors,
//! tied together with a single search entry point.
//!
//! The crux: `search()` traverses the graph using *quantized* distance to
//! the bit codes, then re-ranks the top `ef` survivors with full-precision
//! squared L2. That keeps the inner loop cheap (popcount-style ops on
//! short vectors) while preserving recall via re-ranking.

use crate::graph::{self, Graph};
use crate::quantizer::{self, BitCode, RaBitQuantizer};

#[derive(Clone, Debug)]
pub struct SymphonyQgParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub rotation_seed: u64,
}

impl Default for SymphonyQgParams {
    fn default() -> Self {
        Self { m: 16, ef_construction: 64, ef_search: 64, rotation_seed: 0xA5A5_5A5A }
    }
}

pub struct SymphonyQg {
    pub params: SymphonyQgParams,
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    pub codes: Vec<BitCode>,
    pub graph: Graph,
    pub quantizer: RaBitQuantizer,
}

impl SymphonyQg {
    /// Build the index from a collection of vectors. Full-precision is used
    /// to choose graph edges; quantization is layered on top afterwards.
    pub fn build(vectors: Vec<Vec<f32>>, params: SymphonyQgParams) -> Self {
        assert!(!vectors.is_empty(), "need at least one vector");
        let dim = vectors[0].len();
        for v in &vectors { assert_eq!(v.len(), dim); }

        let quantizer = RaBitQuantizer::new(dim, params.rotation_seed);
        let codes: Vec<BitCode> = vectors.iter().map(|v| quantizer.encode(v)).collect();

        let vectors_ref = &vectors;
        let graph = graph::build_nsw(vectors.len(), params.m, params.ef_construction, |a, b| {
            quantizer::exact_dist_sq(&vectors_ref[a as usize], &vectors_ref[b as usize])
        });

        Self { params, dim, vectors, codes, graph, quantizer }
    }

    /// Quantized-distance graph search + full-precision re-rank.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let eq = self.quantizer.encode_query(query);
        let codes = &self.codes;
        // Graph traversal uses the cheap estimator.
        let coarse = graph::search(
            &self.graph,
            0,
            self.params.ef_search,
            self.params.ef_search,
            |id| quantizer::estimate_dist_sq(&eq, &codes[id as usize]),
        );
        // Re-rank with exact distance.
        let mut rer: Vec<(u32, f32)> = coarse.into_iter().map(|(id, _)| {
            let d = quantizer::exact_dist_sq(query, &self.vectors[id as usize]);
            (id, d)
        }).collect();
        rer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rer.truncate(k);
        rer
    }

    /// SymphonyQG-Fast: symmetric 1-bit popcount distance during traversal,
    /// full-precision re-rank. Much cheaper inner loop than `search()` at
    /// some recall cost.
    pub fn search_popcount(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let eq = self.quantizer.encode_query_symmetric(query);
        let codes = &self.codes;
        let coarse = graph::search(
            &self.graph,
            0,
            self.params.ef_search,
            self.params.ef_search,
            |id| quantizer::estimate_dist_sq_popcount(&eq, &codes[id as usize]),
        );
        let mut rer: Vec<(u32, f32)> = coarse.into_iter().map(|(id, _)| {
            let d = quantizer::exact_dist_sq(query, &self.vectors[id as usize]);
            (id, d)
        }).collect();
        rer.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rer.truncate(k);
        rer
    }

    /// Search using full-precision distance throughout. Useful as a
    /// baseline to measure the recall cost of quantized traversal.
    pub fn search_exact_graph(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let vectors = &self.vectors;
        let coarse = graph::search(
            &self.graph,
            0,
            self.params.ef_search,
            self.params.ef_search,
            |id| quantizer::exact_dist_sq(query, &vectors[id as usize]),
        );
        let mut out: Vec<(u32, f32)> = coarse.into_iter().collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }

    pub fn memory_bytes(&self) -> (usize, usize) {
        let vec_bytes: usize = self.vectors.iter().map(|v| v.len() * 4).sum();
        let code_bytes: usize = self.codes.iter().map(|c| c.nbytes()).sum();
        (vec_bytes, code_bytes)
    }
}

/// Brute-force k-NN. Reference for recall measurements.
pub fn brute_force(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = vectors.iter().enumerate()
        .map(|(i, v)| (i as u32, quantizer::exact_dist_sq(query, v)))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(k);
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0_f32)).collect()).collect()
    }

    #[test]
    fn recall_better_than_random() {
        let n = 2_000;
        let d = 128;
        let k = 10;
        let db = synth(n, d, 42);
        let queries = synth(50, d, 99);

        let idx = SymphonyQg::build(db.clone(), SymphonyQgParams::default());

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let gt: std::collections::HashSet<u32> = brute_force(&db, q, k).into_iter().map(|(i,_)| i).collect();
            let got = idx.search(q, k);
            for (i, _) in got { if gt.contains(&i) { hits += 1; } }
            total += k;
        }
        let recall = hits as f32 / total as f32;
        // PoC bar: significantly better than random (10/2000=0.5%).
        assert!(recall > 0.30, "recall@10 too low: {}", recall);
    }
}
