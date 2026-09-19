//! Deterministic RNG, synthetic dataset generator, brute-force top-k, recall.
use crate::{l2_sq, NodeId, VecStore};
use std::cmp::Ordering;

pub struct SplitMix64(pub u64);
impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    pub fn next_gauss(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-9);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

pub fn make_clustered_dataset(n: usize, dim: usize, k_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64(seed);
    let mut centers = Vec::with_capacity(k_clusters);
    for _ in 0..k_clusters {
        let c: Vec<f32> = (0..dim).map(|_| rng.next_gauss() * 6.0).collect();
        centers.push(c);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let c = &centers[i % k_clusters];
        let v: Vec<f32> = (0..dim).map(|j| c[j] + rng.next_gauss()).collect();
        out.push(v);
    }
    out
}

pub fn brute_force_topk(data: &VecStore, q: &[f32], k: usize) -> Vec<(NodeId, f32)> {
    let mut all: Vec<(NodeId, f32)> = (0..data.n as u32)
        .map(|i| (i, l2_sq(q, data.get(i))))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    all.truncate(k);
    all
}

pub fn recall_at_k(pred: &[(NodeId, f32)], truth: &[(NodeId, f32)], k: usize) -> f32 {
    use std::collections::HashSet;
    let t: HashSet<NodeId> = truth.iter().take(k).map(|(id, _)| *id).collect();
    let hit = pred.iter().take(k).filter(|(id, _)| t.contains(id)).count();
    hit as f32 / k as f32
}
