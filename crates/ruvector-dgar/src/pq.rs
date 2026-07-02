//! Minimal, real (not-mock) product quantizer used to drive the DGAR benchmarks.
//!
//! The goal here is *not* to compete with FAISS-PQ — it is to produce an
//! honest, noisy approximate-distance oracle so that rerankers see a realistic
//! stream of `(id, approx_distance)` tuples.  The quantizer is:
//!
//! * `M` sub-quantizers, each with `K = 256` centroids trained via mini-batch
//!   k-means (`kmeans_iters` passes) on the training vectors.
//! * L2 asymmetric distance computed from precomputed sub-vector LUTs.
//!
//! The implementation is deliberately compact (<200 lines) so the whole
//! benchmark harness stays under the workspace 500-line file cap.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// A raw f32 vector.
pub type Vector = Vec<f32>;

/// A query with its raw vector; the id-space is implicit (caller-managed).
#[derive(Clone, Debug)]
pub struct Query<'a> {
    /// Raw query vector.
    pub vector: &'a [f32],
}

/// One row emitted by the approximate stage: `(id, approx_distance)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApproxCandidate {
    /// Row id of the candidate in the base set.
    pub id: u32,
    /// Approximate squared-L2 distance from the PQ LUT.
    pub distance: f32,
}

/// One row emitted by a reranker: the exact-distance-verified top-K.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RerankResult {
    /// Row id.
    pub id: u32,
    /// Exact squared-L2 distance.
    pub distance: f32,
}

/// Small, honest PQ implementation.  Not tuned for speed — tuned for realism.
#[derive(Serialize, Deserialize)]
pub struct ProductQuantizer {
    m: usize,
    ks: usize,
    dsub: usize,
    /// centroids[m][ks * dsub]
    centroids: Vec<Vec<f32>>,
}

impl ProductQuantizer {
    /// Train `m` sub-quantizers with `ks` centroids each, running
    /// `kmeans_iters` Lloyd passes on `training` (row-major).
    pub fn train(
        training: &[Vector],
        dim: usize,
        m: usize,
        ks: usize,
        kmeans_iters: usize,
        seed: u64,
    ) -> Self {
        assert!(dim % m == 0, "dim must be divisible by m");
        let dsub = dim / m;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut centroids = Vec::with_capacity(m);
        for sub in 0..m {
            let mut c = vec![0.0f32; ks * dsub];
            // init: pick `ks` random training rows for this subspace
            for k in 0..ks {
                let src = &training[rng.gen_range(0..training.len())];
                let off = sub * dsub;
                c[k * dsub..(k + 1) * dsub].copy_from_slice(&src[off..off + dsub]);
            }
            // lloyd
            for _ in 0..kmeans_iters {
                let mut sums = vec![0.0f32; ks * dsub];
                let mut counts = vec![0u32; ks];
                for row in training {
                    let sv = &row[sub * dsub..(sub + 1) * dsub];
                    let (best, _) = nearest(sv, &c, ks, dsub);
                    counts[best] += 1;
                    for j in 0..dsub {
                        sums[best * dsub + j] += sv[j];
                    }
                }
                for k in 0..ks {
                    if counts[k] > 0 {
                        let inv = 1.0 / counts[k] as f32;
                        for j in 0..dsub {
                            c[k * dsub + j] = sums[k * dsub + j] * inv;
                        }
                    }
                }
            }
            centroids.push(c);
        }
        Self {
            m,
            ks,
            dsub,
            centroids,
        }
    }

    /// Encode a raw vector into `m` sub-codes.
    pub fn encode(&self, x: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.m];
        for sub in 0..self.m {
            let sv = &x[sub * self.dsub..(sub + 1) * self.dsub];
            let (best, _) = nearest(sv, &self.centroids[sub], self.ks, self.dsub);
            out[sub] = best as u8;
        }
        out
    }

    /// Precompute the query-conditioned LUT.  `lut[sub * ks + code]` is the
    /// squared-L2 distance between query sub-vector `sub` and centroid `code`.
    pub fn build_lut(&self, query: &[f32]) -> Vec<f32> {
        let mut lut = vec![0.0f32; self.m * self.ks];
        for sub in 0..self.m {
            let qv = &query[sub * self.dsub..(sub + 1) * self.dsub];
            for k in 0..self.ks {
                let c = &self.centroids[sub][k * self.dsub..(k + 1) * self.dsub];
                let mut s = 0.0f32;
                for j in 0..self.dsub {
                    let d = qv[j] - c[j];
                    s += d * d;
                }
                lut[sub * self.ks + k] = s;
            }
        }
        lut
    }

    /// Approximate squared-L2 by summing the sub-vector LUT rows.
    pub fn adc(&self, lut: &[f32], codes: &[u8]) -> f32 {
        let mut s = 0.0f32;
        for sub in 0..self.m {
            s += lut[sub * self.ks + codes[sub] as usize];
        }
        s
    }

    /// Approximate top-N candidate stream, sorted ascending by approx distance.
    pub fn search(&self, query: &[f32], corpus_codes: &[Vec<u8>], n: usize) -> Vec<ApproxCandidate> {
        let lut = self.build_lut(query);
        let mut all: Vec<ApproxCandidate> = corpus_codes
            .iter()
            .enumerate()
            .map(|(i, c)| ApproxCandidate {
                id: i as u32,
                distance: self.adc(&lut, c),
            })
            .collect();
        all.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        all.truncate(n);
        all
    }
}

fn nearest(x: &[f32], centroids: &[f32], ks: usize, dsub: usize) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for k in 0..ks {
        let c = &centroids[k * dsub..(k + 1) * dsub];
        let mut s = 0.0f32;
        for j in 0..dsub {
            let d = x[j] - c[j];
            s += d * d;
        }
        if s < best_d {
            best_d = s;
            best = k;
        }
    }
    (best, best_d)
}

/// Brute-force squared-L2 — the exact-distance oracle used for reranking and
/// ground truth.
pub struct BruteForce<'a> {
    /// Row-major corpus.
    pub corpus: &'a [Vector],
}

impl<'a> BruteForce<'a> {
    /// Exact squared-L2 for one candidate id.
    pub fn distance(&self, query: &[f32], id: u32) -> f32 {
        let row = &self.corpus[id as usize];
        let mut s = 0.0f32;
        for j in 0..row.len() {
            let d = query[j] - row[j];
            s += d * d;
        }
        s
    }

    /// Ground-truth top-K by exhaustive scan.
    pub fn topk(&self, query: &[f32], k: usize) -> Vec<RerankResult> {
        let mut all: Vec<RerankResult> = (0..self.corpus.len() as u32)
            .map(|id| RerankResult {
                id,
                distance: self.distance(query, id),
            })
            .collect();
        all.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        all.truncate(k);
        all
    }
}
