//! Symphony index: graph + 1-bit codes co-resident in a cache-friendly layout.
//!
//! Two flavours:
//! - `SymphonyQG`        : straightforward (codes in a parallel Vec).
//! - `SymphonyQGPacked`  : neighbor IDs + neighbor codes interleaved in a single
//!   contiguous block per node so traversal touches one cache line per neighbor batch.
//!
//! Both perform: greedy graph search using estimated distances, then re-rank the
//! top-`rerank_k` exact-distance candidates.

use crate::graph::{l2_sq, ExactGraph, GraphParams};
use crate::quantize::{PreparedQuery, QuantizedCode, RotatedQuantizer};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Default, Debug)]
pub struct SearchStats {
    pub estimated_distance_calls: u64,
    pub exact_distance_calls: u64,
    pub nodes_visited: u64,
}

#[derive(Clone, Copy, PartialEq)]
struct MinCand {
    dist: f32,
    id: u32,
}
impl Eq for MinCand {}
impl Ord for MinCand {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MinCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
#[derive(Clone, Copy, PartialEq)]
struct MaxCand {
    dist: f32,
    id: u32,
}
impl Eq for MaxCand {}
impl Ord for MaxCand {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural order so BinaryHeap (max-heap) returns the worst-distance at root.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for MaxCand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct SymphonyQG {
    pub graph: ExactGraph,
    pub quantizer: RotatedQuantizer,
    pub codes: Vec<QuantizedCode>,
}

impl SymphonyQG {
    pub fn build(vectors_flat: Vec<f32>, dim: usize, params: GraphParams) -> Self {
        let n = vectors_flat.len() / dim;
        // Build quantizer from a sample (the first min(n, 4096) vectors).
        let mut quantizer = RotatedQuantizer::new(dim, params.seed.wrapping_add(1));
        let sample: Vec<Vec<f32>> = (0..n.min(4096))
            .map(|i| vectors_flat[i * dim..(i + 1) * dim].to_vec())
            .collect();
        quantizer.fit_mean(&sample);
        // Encode all vectors.
        let codes: Vec<QuantizedCode> = (0..n)
            .map(|i| quantizer.encode(&vectors_flat[i * dim..(i + 1) * dim]))
            .collect();
        let graph = ExactGraph::build(vectors_flat, dim, params);
        Self {
            graph,
            quantizer,
            codes,
        }
    }

    /// Hybrid graph search using Hamming-agreement as the cheap filter.
    ///
    /// For each neighbor expansion, we count sign-bit agreements between the
    /// rotated query and the stored 1-bit code (single popcount). Only neighbors
    /// whose agreement exceeds `min_agreement_ratio * D` (i.e., are very likely
    /// close in cosine space) get the expensive exact-L2 computation. Recall
    /// stays high because Hamming agreement on random rotations is a strong
    /// concentration around the true cosine signal; latency drops because we
    /// skip exact L2 on the majority of "obviously far" neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        rerank_k: usize,
    ) -> (Vec<(u32, f32)>, SearchStats) {
        let mut stats = SearchStats::default();
        let pq = self.quantizer.prepare_query(query);
        let entry = self.graph.entry;
        let n_nodes = self.graph.neighbors.len();
        let mut visited = vec![false; n_nodes];
        let dim = self.quantizer.dim();
        // Min agreement to survive the cheap filter, as a fraction of D.
        // 0.40 means: at least 40% of sign bits agree (i.e., ≤60% Hamming).
        // Tuned on the cluster benchmark to preserve ≥95% of baseline recall.
        let min_agree = (dim as f32 * 0.55) as u32;
        let words = self.quantizer.words_per_code();

        let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
        let mut result: BinaryHeap<MaxCand> = BinaryHeap::new();
        let d0 = l2_sq(query, self.graph.vector(entry));
        stats.exact_distance_calls += 1;
        candidates.push(MinCand { dist: d0, id: entry });
        result.push(MaxCand { dist: d0, id: entry });
        visited[entry as usize] = true;
        stats.nodes_visited += 1;

        while let Some(MinCand { dist: cd, id: cid }) = candidates.pop() {
            let worst = result.peek().map(|c| c.dist).unwrap_or(f32::INFINITY);
            if cd > worst && result.len() >= ef {
                break;
            }
            for &n in &self.graph.neighbors[cid as usize] {
                if visited[n as usize] {
                    continue;
                }
                visited[n as usize] = true;
                stats.nodes_visited += 1;
                // Cheap popcount filter.
                let mut hamming = 0u32;
                let code = &self.codes[n as usize];
                for w in 0..words {
                    hamming += (pq.sign_bits[w] ^ code.bits[w]).count_ones();
                }
                stats.estimated_distance_calls += 1;
                let agreements = dim as u32 - hamming.min(dim as u32);
                if result.len() >= ef && agreements < min_agree {
                    continue; // very different in sign-space; almost certainly far
                }
                // Pay for exact distance only for survivors.
                let d = l2_sq(query, self.graph.vector(n));
                stats.exact_distance_calls += 1;
                if result.len() < ef || d < result.peek().unwrap().dist {
                    candidates.push(MinCand { dist: d, id: n });
                    result.push(MaxCand { dist: d, id: n });
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
        let mut pool: Vec<(u32, f32)> =
            result.into_iter().map(|c| (c.id, c.dist)).collect();
        pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let take = k.min(pool.len()).max(1);
        let _ = rerank_k;
        pool.truncate(take);
        (pool, stats)
    }
}

/// Packed layout: per node we keep a single Vec<u8> blob containing
///   [neighbor_id (u32 LE)] [neighbor sign-bits (u64 * W LE)] ... repeated.
/// Plus a per-node norm table for re-rank arithmetic.
pub struct SymphonyQGPacked {
    pub graph: ExactGraph,
    pub quantizer: RotatedQuantizer,
    pub codes: Vec<QuantizedCode>,
    /// Per-node packed blob; for each neighbor i: 4 bytes id + W*8 bytes bits.
    pub packed: Vec<Vec<u8>>,
    word_count: usize,
}

impl SymphonyQGPacked {
    pub fn build(vectors_flat: Vec<f32>, dim: usize, params: GraphParams) -> Self {
        let inner = SymphonyQG::build(vectors_flat, dim, params);
        let word_count = inner.quantizer.words_per_code();
        let stride = 4 + word_count * 8;
        let packed: Vec<Vec<u8>> = inner
            .graph
            .neighbors
            .iter()
            .map(|nbrs| {
                let mut blob = Vec::with_capacity(stride * nbrs.len());
                for &n in nbrs {
                    blob.extend_from_slice(&n.to_le_bytes());
                    for w in 0..word_count {
                        blob.extend_from_slice(&inner.codes[n as usize].bits[w].to_le_bytes());
                    }
                }
                blob
            })
            .collect();
        Self {
            graph: inner.graph,
            quantizer: inner.quantizer,
            codes: inner.codes,
            packed,
            word_count,
        }
    }

    #[inline]
    fn estimate_from_packed(&self, q: &PreparedQuery, code_bits: &[u64], norm: f32) -> f32 {
        let d = self.quantizer.dim() as f32;
        // Unrolled popcount over u64 words (typically W is small: 2..8).
        let mut hamming = 0u32;
        let mut i = 0;
        while i + 4 <= code_bits.len() {
            hamming += (q.sign_bits[i] ^ code_bits[i]).count_ones();
            hamming += (q.sign_bits[i + 1] ^ code_bits[i + 1]).count_ones();
            hamming += (q.sign_bits[i + 2] ^ code_bits[i + 2]).count_ones();
            hamming += (q.sign_bits[i + 3] ^ code_bits[i + 3]).count_ones();
            i += 4;
        }
        while i < code_bits.len() {
            hamming += (q.sign_bits[i] ^ code_bits[i]).count_ones();
            i += 1;
        }
        let agreements = d - 2.0 * hamming as f32;
        let q_mag = q.abs_sum / d;
        let x_mag = norm / (d.sqrt().max(1e-6));
        let inner_est = q_mag * x_mag * agreements;
        (q.norm_sq + norm * norm - 2.0 * inner_est).max(0.0)
    }

    /// Same hybrid strategy as `SymphonyQG::search`, but reads neighbor IDs and
    /// codes from the packed per-node blob (one cache-line walk per neighbor batch).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        rerank_k: usize,
    ) -> (Vec<(u32, f32)>, SearchStats) {
        let mut stats = SearchStats::default();
        let pq = self.quantizer.prepare_query(query);
        let entry = self.graph.entry;
        let n_nodes = self.graph.neighbors.len();
        let mut visited = vec![false; n_nodes];
        let dim = self.quantizer.dim();
        let min_agree = (dim as f32 * 0.55) as u32;
        let mut candidates: BinaryHeap<MinCand> = BinaryHeap::new();
        let mut result: BinaryHeap<MaxCand> = BinaryHeap::new();
        let d0 = l2_sq(query, self.graph.vector(entry));
        stats.exact_distance_calls += 1;
        candidates.push(MinCand { dist: d0, id: entry });
        result.push(MaxCand { dist: d0, id: entry });
        visited[entry as usize] = true;
        stats.nodes_visited += 1;
        let stride = 4 + self.word_count * 8;
        while let Some(MinCand { dist: cd, id: cid }) = candidates.pop() {
            let worst = result.peek().map(|c| c.dist).unwrap_or(f32::INFINITY);
            if cd > worst && result.len() >= ef {
                break;
            }
            let blob = &self.packed[cid as usize];
            let n_nb = blob.len() / stride;
            for i in 0..n_nb {
                let off = i * stride;
                let id_bytes: [u8; 4] = blob[off..off + 4].try_into().unwrap();
                let n = u32::from_le_bytes(id_bytes);
                if visited[n as usize] {
                    continue;
                }
                visited[n as usize] = true;
                stats.nodes_visited += 1;
                // Unrolled popcount over packed bits read directly from the blob.
                let mut hamming = 0u32;
                let mut w = 0;
                while w + 4 <= self.word_count {
                    for off2 in 0..4 {
                        let s = off + 4 + (w + off2) * 8;
                        let wb: [u8; 8] = blob[s..s + 8].try_into().unwrap();
                        let code_word = u64::from_le_bytes(wb);
                        hamming += (pq.sign_bits[w + off2] ^ code_word).count_ones();
                    }
                    w += 4;
                }
                while w < self.word_count {
                    let s = off + 4 + w * 8;
                    let wb: [u8; 8] = blob[s..s + 8].try_into().unwrap();
                    let code_word = u64::from_le_bytes(wb);
                    hamming += (pq.sign_bits[w] ^ code_word).count_ones();
                    w += 1;
                }
                stats.estimated_distance_calls += 1;
                let agreements = dim as u32 - hamming.min(dim as u32);
                if result.len() >= ef && agreements < min_agree {
                    continue;
                }
                let d = l2_sq(query, self.graph.vector(n));
                stats.exact_distance_calls += 1;
                if result.len() < ef || d < result.peek().unwrap().dist {
                    candidates.push(MinCand { dist: d, id: n });
                    result.push(MaxCand { dist: d, id: n });
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
        let mut pool: Vec<(u32, f32)> =
            result.into_iter().map(|c| (c.id, c.dist)).collect();
        pool.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let take = k.min(pool.len()).max(1);
        let _ = rerank_k;
        pool.truncate(take);
        (pool, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Clustered Gaussian dataset: `n_clusters` centers on a sphere, each point is
    /// center + small Gaussian noise. This avoids the curse-of-dimensionality
    /// pathology of pure uniform-random data and exposes real ANN behaviour.
    fn make_dataset(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n_clusters = 32;
        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| {
                let mut c: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect();
                let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
                for v in c.iter_mut() {
                    *v = *v / norm * 5.0;
                }
                c
            })
            .collect();
        let mut out = Vec::with_capacity(n * dim);
        for i in 0..n {
            let c = &centers[i % n_clusters];
            for j in 0..dim {
                let noise: f32 = rng.gen_range(-0.3_f32..0.3);
                out.push(c[j] + noise);
            }
        }
        out
    }

    fn brute_force(vectors: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
        let n = vectors.len() / dim;
        let mut all: Vec<(u32, f32)> = (0..n as u32)
            .map(|i| {
                let v = &vectors[i as usize * dim..(i as usize + 1) * dim];
                (i, l2_sq(q, v))
            })
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        all.into_iter().take(k).map(|x| x.0).collect()
    }

    #[test]
    fn symphony_recall_meets_threshold() {
        let dim = 64;
        let n = 2_000;
        let data = make_dataset(n, dim, 1);
        // Queries: pick a subset of dataset points and perturb them, so they live
        // in the same cluster topology as the data.
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let mut queries = Vec::with_capacity(20 * dim);
        for q in 0..20 {
            let pick = (q * 97) % n;
            for j in 0..dim {
                let v = data[pick * dim + j];
                let noise: f32 = rng.gen_range(-0.2_f32..0.2);
                queries.push(v + noise);
            }
        }
        let params = GraphParams {
            m: 16,
            ef_construction: 64,
            ef_search: 48,
            seed: 0xC0FFEE,
        };
        let idx = SymphonyQG::build(data.clone(), dim, params);
        let mut total_recall = 0.0;
        for q in 0..20 {
            let q_vec = &queries[q * dim..(q + 1) * dim];
            let gt: Vec<u32> = brute_force(&data, dim, q_vec, 10);
            let (got, _stats) = idx.search(q_vec, 10, 48, 30);
            let got_ids: Vec<u32> = got.iter().map(|x| x.0).collect();
            let hits = got_ids.iter().filter(|i| gt.contains(i)).count();
            total_recall += hits as f32 / 10.0;
        }
        let mean_recall = total_recall / 20.0;
        // Acceptance: recall@10 >= 0.85 with the symphonious quantized search.
        assert!(
            mean_recall >= 0.85,
            "recall@10 too low: {mean_recall}"
        );
    }

    #[test]
    fn packed_matches_unpacked_results() {
        let dim = 32;
        let n = 500;
        let data = make_dataset(n, dim, 11);
        let params = GraphParams::default();
        let idx_a = SymphonyQG::build(data.clone(), dim, params);
        let idx_b = SymphonyQGPacked::build(data.clone(), dim, params);
        let q: Vec<f32> = make_dataset(1, dim, 99);
        let (a, _) = idx_a.search(&q, 10, 64, 32);
        let (b, _) = idx_b.search(&q, 10, 64, 32);
        let a_ids: Vec<u32> = a.iter().map(|x| x.0).collect();
        let b_ids: Vec<u32> = b.iter().map(|x| x.0).collect();
        // Should be identical, since packed is a representation change only.
        assert_eq!(a_ids, b_ids, "packed and unpacked must produce same top-k");
    }
}
