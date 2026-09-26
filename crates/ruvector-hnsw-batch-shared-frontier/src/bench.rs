//! Benchmark harness: builds a synthetic clustered dataset, generates a
//! correlated query batch, and measures the three backends end-to-end.

use crate::batched::BatchedKernelSearch;
use crate::graph::{ProximityGraph, ProximityGraphBuilder};
use crate::independent::IndependentSearch;
use crate::shared::SharedFrontierSearch;
use crate::TopK;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// Structured output for a single backend run.
#[derive(Debug, Clone)]
pub struct BackendResult {
    pub name: &'static str,
    pub elapsed_ms: f64,
    pub loads: u64,
    pub cache_hits: u64,
    /// Mean recall @ k against exact brute-force ground truth.
    pub recall: f64,
}

/// Full benchmark run summary.
#[derive(Debug, Clone)]
pub struct BenchReport {
    pub n: usize,
    pub dim: usize,
    pub q: usize,
    pub k: usize,
    pub ef: usize,
    pub batch_correlation: f32,
    pub results: Vec<BackendResult>,
}

pub fn synth_clustered_dataset(
    n: usize,
    dim: usize,
    num_clusters: usize,
    seed: u64,
) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let c = &centers[i % num_clusters];
        for d in 0..dim {
            out[i * dim + d] = c[d] + rng.gen_range(-0.15..0.15);
        }
    }
    out
}

/// Build a correlated query batch: pick one anchor query then perturb it
/// with noise scaled by `1.0 - corr`. If `anchor_from` is `Some`, the
/// anchor is drawn from an existing vector in that dataset instead of
/// random space, which keeps recall dominated by graph-traversal quality
/// rather than by whether the anchor happens to land near any cluster.
pub fn synth_correlated_queries(
    q: usize,
    dim: usize,
    corr: f32,
    seed: u64,
    anchor_from: Option<&[f32]>,
) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let anchor: Vec<f32> = if let Some(data) = anchor_from {
        let n = data.len() / dim;
        let pick = rng.gen_range(0..n);
        data[pick * dim..(pick + 1) * dim].to_vec()
    } else {
        (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    };
    // "corr" scales perturbation on top of the anchor. The perturbation
    // magnitude is chosen so a batch of correlation 1.0 collapses to the
    // anchor and 0.0 spreads with unit noise.
    let noise = 1.0 - corr;
    (0..q).map(|_| {
        anchor.iter().map(|&a| a + rng.gen_range(-noise..noise)).collect()
    }).collect()
}

fn recall_at_k(got: &[TopK], truth: &[Vec<u32>]) -> f64 {
    assert_eq!(got.len(), truth.len());
    let mut hit = 0.0f64;
    let mut denom = 0.0f64;
    for (g, t) in got.iter().zip(truth.iter()) {
        let ts: std::collections::HashSet<u32> = t.iter().copied().collect();
        for id in &g.ids {
            if ts.contains(id) { hit += 1.0; }
        }
        denom += t.len() as f64;
    }
    if denom > 0.0 { hit / denom } else { 0.0 }
}

pub fn run_bench(
    n: usize, dim: usize, q: usize, k: usize, ef: usize,
    correlation: f32, seed: u64,
) -> BenchReport {
    let data = synth_clustered_dataset(n, dim, 32, seed);
    let queries = synth_correlated_queries(
        q, dim, correlation, seed.wrapping_add(2), Some(&data),
    );
    let graph: ProximityGraph = ProximityGraphBuilder {
        n, dim, m: 32, pool: 256, seed: seed.wrapping_add(1),
    }.build(data);

    // Ground truth via brute force.
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| graph.brute_top_k(q, k)).collect();

    // Independent baseline.
    let t = Instant::now();
    let mut loads_ind: u64 = 0;
    let mut out_ind = Vec::with_capacity(q);
    for query in &queries {
        let (r, l) = IndependentSearch::search_one(&graph, query, k, ef);
        loads_ind += l;
        out_ind.push(r);
    }
    let ind = BackendResult {
        name: "independent",
        elapsed_ms: t.elapsed().as_secs_f64() * 1e3,
        loads: loads_ind,
        cache_hits: 0,
        recall: recall_at_k(&out_ind, &truth),
    };

    // Batched-kernel.
    let t = Instant::now();
    let (out_bk, loads_bk) = BatchedKernelSearch::search(&graph, &queries, k, ef);
    let bk = BackendResult {
        name: "batched-kernel",
        elapsed_ms: t.elapsed().as_secs_f64() * 1e3,
        loads: loads_bk,
        cache_hits: 0,
        recall: recall_at_k(&out_bk, &truth),
    };

    // Shared-frontier.
    let t = Instant::now();
    let (out_sf, loads_sf, hits_sf) = SharedFrontierSearch::search(&graph, &queries, k, ef);
    let sf = BackendResult {
        name: "shared-frontier",
        elapsed_ms: t.elapsed().as_secs_f64() * 1e3,
        loads: loads_sf,
        cache_hits: hits_sf,
        recall: recall_at_k(&out_sf, &truth),
    };

    BenchReport {
        n, dim, q, k, ef,
        batch_correlation: correlation,
        results: vec![ind, bk, sf],
    }
}

impl BenchReport {
    /// Estimated peak memory of the shared-frontier cache in bytes:
    /// one Q-length f32 row per graph node.
    pub fn shared_cache_bytes(&self) -> usize {
        self.n * self.q * std::mem::size_of::<f32>()
    }

    pub fn print(&self) {
        println!("=== BenchReport n={} dim={} q={} k={} ef={} corr={:.2} ===",
                 self.n, self.dim, self.q, self.k, self.ef, self.batch_correlation);
        println!("{:<18} {:>10} {:>10} {:>10} {:>8}", "backend", "ms", "loads", "hits", "recall");
        for r in &self.results {
            println!("{:<18} {:>10.2} {:>10} {:>10} {:>8.4}",
                r.name, r.elapsed_ms, r.loads, r.cache_hits, r.recall);
        }
        let base_loads = self.results[0].loads as f64;
        let sf_loads = self.results[2].loads as f64;
        let save_pct = 100.0 * (1.0 - sf_loads / base_loads);
        println!("shared-frontier vector-load savings vs independent: {:.1}%", save_pct);
        println!("shared-frontier cache footprint estimate: {} bytes ({:.2} MiB)",
                 self.shared_cache_bytes(),
                 self.shared_cache_bytes() as f64 / (1024.0 * 1024.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BatchedSearch;

    #[test]
    fn all_backends_agree_on_recall() {
        let report = run_bench(400, 32, 8, 5, 32, 0.85, 7);
        for r in &report.results {
            assert!(r.recall > 0.5, "backend {} recall too low: {}", r.name, r.recall);
        }
    }

    #[test]
    fn shared_saves_loads_on_correlated_batch() {
        let report = run_bench(600, 32, 16, 10, 48, 0.95, 11);
        let ind = &report.results[0];
        let sf = &report.results[2];
        assert!(sf.loads < ind.loads,
            "shared-frontier loads {} should undercut independent {}",
            sf.loads, ind.loads);
    }

    #[test]
    fn trait_object_usable() {
        use crate::independent::IndependentSearch;
        let data = synth_clustered_dataset(200, 16, 4, 3);
        let g = crate::graph::ProximityGraphBuilder { n: 200, dim: 16, m: 8, pool: 64, seed: 4 }.build(data);
        let queries = synth_correlated_queries(4, 16, 0.9, 5, None);
        let backend: &dyn BatchedSearch = &IndependentSearch;
        let out = backend.search_batch(&g, &queries, 3, 16);
        assert_eq!(out.len(), 4);
        for t in out { assert_eq!(t.ids.len(), 3); }
    }
}
