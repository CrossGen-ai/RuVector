//! SymphonyQG-style coupled graph + binary-quantization search.
//!
//! Three components live together:
//!   1. A kNN graph (substrate, built once with full f32).
//!   2. A packed binary code table (`Vec<u64>`, n × words) — one row per node.
//!   3. A search loop that uses Hamming distance to rank candidates and
//!      reranks the top-`rerank` survivors with f32 only at the end.
//!
//! The coupling is what matters: the candidate score during graph
//! traversal is cheap (popcount over a handful of u64), so we can afford
//! a wide `ef_search` while keeping wall-clock cost low.

use crate::{l2_sq, quant::BitQuantizer, graph::{KnnGraph, GraphParams}};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Clone, Debug)]
pub struct IndexParams {
    pub graph: GraphParams,
    pub quant_seed: u64,
    /// During search, top-`rerank` Hamming candidates are reranked with f32.
    pub rerank: usize,
}

impl Default for IndexParams {
    fn default() -> Self {
        Self { graph: GraphParams::default(), quant_seed: 0xA5A5_A5A5, rerank: 64 }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SearchStats {
    pub visited: u32,
    pub f32_ops: u32,
    pub ham_ops: u32,
}

pub struct SymphonyIndex {
    pub d: usize,
    pub graph: KnnGraph,
    pub quant: BitQuantizer,
    pub codes: Vec<u64>,       // n * quant.words
    pub vectors: Vec<Vec<f32>>, // owned copy for reranking
    pub params: IndexParams,
}

impl SymphonyIndex {
    pub fn build(vectors: &[Vec<f32>], params: IndexParams) -> Self {
        assert!(!vectors.is_empty(), "empty corpus");
        let d = vectors[0].len();
        let graph = KnnGraph::build(vectors, params.graph.clone());
        let quant = BitQuantizer::new(d, params.quant_seed);
        let codes = quant.encode_batch(vectors);
        Self { d, graph, quant, codes, vectors: vectors.to_vec(), params }
    }

    /// Memory footprint estimate in bytes.
    pub fn estimated_bytes(&self) -> usize {
        let v = self.vectors.len() * self.d * std::mem::size_of::<f32>();
        let g = self.graph.adj.len() * std::mem::size_of::<u32>();
        let c = self.codes.len() * std::mem::size_of::<u64>();
        v + g + c
    }

    /// Symphony search: traverse graph using Hamming candidate ranking, then
    /// rerank the top `params.rerank` survivors with full f32 distances.
    pub fn search(&self, query: &[f32], topk: usize) -> (Vec<(usize, f32)>, SearchStats) {
        self.search_with_ef(query, topk, self.params.graph.ef_search)
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        topk: usize,
        ef: usize,
    ) -> (Vec<(usize, f32)>, SearchStats) {
        let qcode = self.quant.encode(query);
        let mut stats = SearchStats::default();

        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF);
        let entry = rng.gen_range(0..self.graph.n);

        let mut visited = vec![false; self.graph.n];
        // Frontier: BinaryHeap min-heap on hamming
        let mut heap: std::collections::BinaryHeap<NegU32> = Default::default();
        let mut results: std::collections::BinaryHeap<PosU32> = Default::default();

        let h0 = self.quant.hamming_at(&qcode, &self.codes, entry);
        stats.ham_ops += 1;
        heap.push(NegU32(h0, entry as u32));
        results.push(PosU32(h0, entry as u32));
        visited[entry] = true;
        stats.visited += 1;

        while let Some(NegU32(h, node)) = heap.pop() {
            if let Some(PosU32(top, _)) = results.peek() {
                if results.len() >= ef && h > *top { break; }
            }
            for &nb in self.graph.neighbours(node as usize) {
                let nbu = nb as usize;
                if visited[nbu] { continue; }
                visited[nbu] = true;
                stats.visited += 1;
                let hn = self.quant.hamming_at(&qcode, &self.codes, nbu);
                stats.ham_ops += 1;

                if results.len() < ef {
                    heap.push(NegU32(hn, nb));
                    results.push(PosU32(hn, nb));
                } else if let Some(PosU32(top, _)) = results.peek() {
                    if hn < *top {
                        heap.push(NegU32(hn, nb));
                        results.push(PosU32(hn, nb));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }

        // Rerank top `rerank` (by hamming) with f32 distance.
        let mut survivors: Vec<u32> = results.into_iter().map(|PosU32(_, i)| i).collect();
        survivors.sort_unstable();
        survivors.dedup();
        if survivors.len() > self.params.rerank {
            // hamming-keyed truncation: recompute ham to sort (cheap), keep best `rerank`
            let mut s2: Vec<(u32, u32)> = survivors.iter()
                .map(|&i| (self.quant.hamming_at(&qcode, &self.codes, i as usize), i))
                .collect();
            stats.ham_ops += s2.len() as u32;
            s2.sort_by_key(|x| x.0);
            survivors = s2.into_iter().take(self.params.rerank).map(|x| x.1).collect();
        }

        let mut reranked: Vec<(usize, f32)> = survivors.iter()
            .map(|&i| {
                let iu = i as usize;
                (iu, l2_sq(&self.vectors[iu], query))
            })
            .collect();
        stats.f32_ops += reranked.len() as u32;
        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        reranked.truncate(topk);
        (reranked, stats)
    }
}

#[derive(PartialEq, Eq)]
struct PosU32(u32, u32);
impl Ord for PosU32 { fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.cmp(&other.0) } }
impl PartialOrd for PosU32 { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }

#[derive(PartialEq, Eq)]
struct NegU32(u32, u32);
impl Ord for NegU32 { fn cmp(&self, other: &Self) -> std::cmp::Ordering { other.0.cmp(&self.0) } }
impl PartialOrd for NegU32 { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute_force_knn;

    fn fake_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| {
            let v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
            let n = v.iter().map(|x| x*x).sum::<f32>().sqrt().max(1e-9);
            v.iter().map(|x| x / n).collect()
        }).collect()
    }

    #[test]
    fn symphony_recall_reasonable() {
        let data = fake_data(2000, 64, 17);
        let idx = SymphonyIndex::build(&data, IndexParams {
            graph: GraphParams { k: 32, iters: 4, ef_search: 96, seed: 2 },
            quant_seed: 99,
            rerank: 64,
        });
        let q = &data[42];
        let truth = brute_force_knn(&data, q, 10);
        let (hits, _) = idx.search(q, 10);
        let r = crate::recall_at_k(&hits, &truth);
        // PoC bar — recall must beat random by a wide margin
        assert!(r >= 0.5, "recall too low: {}", r);
    }
}
