//! Product-Quantization asymmetric distance rerank — strong baseline.
//!
//! Vectors are split into `n_sub` subvectors; each subspace is clustered
//! into `k_centroids` centroids by Lloyd's algorithm. At query time a
//! distance table is precomputed so each candidate rerank is `n_sub`
//! table lookups + adds (ADC).

use crate::{sq_l2, Reranker};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::time::Instant;

pub struct PqReranker {
    pub n_sub: usize,
    pub sub_dim: usize,
    pub k_centroids: usize,
    pub codebooks: Vec<Vec<Vec<f32>>>, // [n_sub][k_centroids][sub_dim]
    pub codes: Vec<Vec<u8>>,           // [N][n_sub], u8 → k<=256
}

impl PqReranker {
    pub fn build(
        vectors: &[Vec<f32>],
        n_sub: usize,
        k_centroids: usize,
        lloyd_iters: usize,
        seed: u64,
    ) -> (Self, std::time::Duration) {
        assert!(k_centroids <= 256, "codes stored in u8");
        let start = Instant::now();
        let dim = vectors[0].len();
        assert!(dim % n_sub == 0, "dim must divide by n_sub");
        let sub_dim = dim / n_sub;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(n_sub);

        for s in 0..n_sub {
            // Slice sub-vectors.
            let subs: Vec<&[f32]> = vectors
                .iter()
                .map(|v| &v[s * sub_dim..(s + 1) * sub_dim])
                .collect();
            // Init k-means++: start with random picks (simpler than full k-means++).
            let mut idx: Vec<usize> = (0..subs.len()).collect();
            idx.shuffle(&mut rng);
            let mut centroids: Vec<Vec<f32>> = idx
                .iter()
                .take(k_centroids)
                .map(|&i| subs[i].to_vec())
                .collect();

            for _ in 0..lloyd_iters {
                let mut sums: Vec<Vec<f32>> = vec![vec![0f32; sub_dim]; k_centroids];
                let mut counts: Vec<u32> = vec![0; k_centroids];
                for v in subs.iter() {
                    let mut best_c = 0usize;
                    let mut best_d = f32::INFINITY;
                    for (ci, c) in centroids.iter().enumerate() {
                        let d = sq_l2(v, c);
                        if d < best_d {
                            best_d = d;
                            best_c = ci;
                        }
                    }
                    for (j, x) in v.iter().enumerate() {
                        sums[best_c][j] += *x;
                    }
                    counts[best_c] += 1;
                }
                for ci in 0..k_centroids {
                    if counts[ci] > 0 {
                        let inv = 1.0 / counts[ci] as f32;
                        for j in 0..sub_dim {
                            centroids[ci][j] = sums[ci][j] * inv;
                        }
                    }
                }
            }
            codebooks.push(centroids);
        }

        // Encode all vectors.
        let codes: Vec<Vec<u8>> = vectors
            .iter()
            .map(|v| encode(v, &codebooks, n_sub, sub_dim))
            .collect();

        let elapsed = start.elapsed();
        (
            Self {
                n_sub,
                sub_dim,
                k_centroids,
                codebooks,
                codes,
            },
            elapsed,
        )
    }

    fn precompute_lut(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut lut: Vec<Vec<f32>> = Vec::with_capacity(self.n_sub);
        for s in 0..self.n_sub {
            let qsub = &query[s * self.sub_dim..(s + 1) * self.sub_dim];
            let row: Vec<f32> = self.codebooks[s]
                .iter()
                .map(|c| sq_l2(qsub, c))
                .collect();
            lut.push(row);
        }
        lut
    }
}

fn encode(v: &[f32], cbs: &[Vec<Vec<f32>>], n_sub: usize, sub_dim: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_sub);
    for s in 0..n_sub {
        let sv = &v[s * sub_dim..(s + 1) * sub_dim];
        let mut best_c = 0u8;
        let mut best_d = f32::INFINITY;
        for (ci, c) in cbs[s].iter().enumerate() {
            let d = sq_l2(sv, c);
            if d < best_d {
                best_d = d;
                best_c = ci as u8;
            }
        }
        out.push(best_c);
    }
    out
}

impl Reranker for PqReranker {
    fn name(&self) -> &'static str {
        "PQ-ADC"
    }
    fn bytes_per_vector(&self) -> usize {
        self.n_sub // u8 code per subspace
    }
    fn rerank(&self, query: &[f32], candidates: &[u32], k: usize) -> Vec<u32> {
        let lut = self.precompute_lut(query);
        let mut scored: Vec<(f32, u32)> = candidates
            .iter()
            .map(|&id| {
                let code = &self.codes[id as usize];
                let mut d = 0f32;
                for s in 0..self.n_sub {
                    d += lut[s][code[s] as usize];
                }
                (d, id)
            })
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.into_iter().take(k).map(|(_, id)| id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_rerank_recovers_near_cluster() {
        let mut vs: Vec<Vec<f32>> = (0..64)
            .map(|i| {
                vec![
                    (i as f32) * 0.001,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            })
            .collect();
        vs.extend((0..64).map(|i| {
            vec![
                5.0 + (i as f32) * 0.001,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
                5.0,
            ]
        }));
        let (pq, _) = PqReranker::build(&vs, 4, 8, 12, 11);
        let query = vec![0.0; 8];
        let cands: Vec<u32> = (0..128u32).collect();
        let top = pq.rerank(&query, &cands, 8);
        assert!(top.iter().all(|&id| id < 64));
    }
}
