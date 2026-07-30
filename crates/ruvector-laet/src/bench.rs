//! Deterministic synthetic-dataset benchmark harness.

use crate::graph::{dist_calls, reset_dist_calls};
use crate::{recall_at_k, Index, SearchStrategy};
use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Clustered synthetic dataset — 32 Gaussian blobs. Much more realistic for ANN than
/// uniform noise (uniform 64-D is essentially incompressible; HNSW recall stays low).
pub fn make_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_clusters = 32usize;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 6.0 - 3.0).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            (0..dim)
                .map(|d| c[d] + (rng.gen::<f32>() - 0.5) * 0.6)
                .collect()
        })
        .collect()
}

/// Queries drawn near cluster centers (mirrors typical retrieval workload).
pub fn make_queries(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xA5A5);
    let n_clusters = 32usize;
    // Re-derive the same centers deterministically from the dataset seed 42.
    let mut c_rng = ChaCha8Rng::seed_from_u64(42);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| c_rng.gen::<f32>() * 6.0 - 3.0).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            (0..dim)
                .map(|d| c[d] + (rng.gen::<f32>() - 0.5) * 1.0)
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct BenchRow {
    pub name: String,
    pub recall_at_10: f32,
    pub avg_dist_calls: f64,
    pub avg_latency_us: f64,
}

pub fn run_strategy<S: SearchStrategy>(
    label: &str,
    index: &Index,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    entry_ids: &[u32],
    k: usize,
    mut strat: S,
) -> BenchRow {
    // Warm-up.
    for q in queries.iter().take(5) {
        let _ = index.search(q, k, entry_ids, &mut strat);
    }
    let mut total_recall = 0f32;
    let mut total_calls = 0u64;
    let mut total_us = 0u128;
    for (q, truth) in queries.iter().zip(truths.iter()) {
        reset_dist_calls();
        let t = Instant::now();
        let (pred, _) = index.search(q, k, entry_ids, &mut strat);
        total_us += t.elapsed().as_micros();
        total_calls += dist_calls();
        total_recall += recall_at_k(&pred, truth);
    }
    let n = queries.len() as f64;
    BenchRow {
        name: label.to_string(),
        recall_at_10: total_recall / queries.len() as f32,
        avg_dist_calls: total_calls as f64 / n,
        avg_latency_us: total_us as f64 / n,
    }
}
