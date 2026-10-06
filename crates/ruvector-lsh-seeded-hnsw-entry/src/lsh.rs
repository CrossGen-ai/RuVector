//! Signed random-hyperplane LSH, used only to pick an entry node.
//!
//! For `b` bits we draw `b` i.i.d. unit-normal hyperplanes H_1..H_b of
//! dimension `d`. The signature of a vector v is a bitstring where bit k
//! is 1 iff <H_k, v> >= 0. The collision probability between two
//! unit-norm vectors with angle θ is 1 - θ/π, so similar directions share
//! codes with high probability. We bucket all build-time nodes by their
//! code and keep one representative per bucket: the node whose vector is
//! closest to the bucket's centroid (which, for a sign-code bucket, is
//! approximated by the first-inserted node — the choice is a lower-order
//! effect since we only need "roughly local", not optimal).

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use std::collections::HashMap;

use crate::graph::Nsw;

pub struct LshSeedIndex {
    pub bits: u8,
    pub dim: usize,
    pub planes: Vec<f32>, // row-major, bits * dim
    pub buckets: HashMap<u64, u32>, // code -> representative node id
    pub fallback: u32,
}

impl LshSeedIndex {
    /// Build from a finished NSW graph.
    pub fn build(index: &Nsw, bits: u8, seed: u64) -> Self {
        assert!(bits <= 64, "code packed into u64");
        let dim = index.dim;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut planes = Vec::with_capacity(bits as usize * dim);
        for _ in 0..(bits as usize * dim) {
            let x: f32 = rng.sample(StandardNormal);
            planes.push(x);
        }
        let mut buckets: HashMap<u64, u32> = HashMap::new();
        for id in 0..index.n as u32 {
            let code = signature(index.vector(id), &planes, bits, dim);
            buckets.entry(code).or_insert(id);
        }
        Self { bits, dim, planes, buckets, fallback: index.fixed_entry }
    }

    /// Choose a seed node for this query. Falls back to the fixed enterpoint
    /// when the bucket is empty, so recall is bounded below by baseline.
    pub fn seed_for(&self, query: &[f32]) -> u32 {
        let code = signature(query, &self.planes, self.bits, self.dim);
        *self.buckets.get(&code).unwrap_or(&self.fallback)
    }

    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }
}

#[inline]
pub(crate) fn signature(v: &[f32], planes: &[f32], bits: u8, dim: usize) -> u64 {
    let mut code: u64 = 0;
    for k in 0..bits as usize {
        let plane = &planes[k * dim..(k + 1) * dim];
        let mut dot = 0.0f32;
        for i in 0..dim {
            dot += plane[i] * v[i];
        }
        if dot >= 0.0 {
            code |= 1u64 << k;
        }
    }
    code
}
