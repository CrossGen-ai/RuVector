//! Bench harness shared by the `finger-bench` binary and the criterion
//! benchmark. Produces real numbers — no mocks, no synthetic results.
//!
//! All datasets are deterministic-seeded synthetic Gaussian vectors;
//! using public ANN benchmark datasets (SIFT/GIST/LAION) is left to a
//! follow-up PR because they require network downloads incompatible
//! with the nightly research sandbox.

use crate::*;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub n: usize,
    pub d: usize,
    pub q: usize,
    pub k: usize,
    pub seed_data: u64,
    pub seed_query: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            n: 10_000,
            d: 128,
            q: 100,
            k: 10,
            seed_data: 0xC0FFEE,
            seed_query: 0xBADF00D,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VariantResult {
    pub name: String,
    pub mean_us_per_query: f64,
    pub recall_at_k: f64,
    pub flops_per_estimate: usize,
}

pub fn make_gaussian(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    let s: f32 = StandardNormal.sample(&mut rng);
                    s
                })
                .collect()
        })
        .collect()
}

pub fn run_full_bench(cfg: &BenchConfig) -> Vec<VariantResult> {
    let base = make_gaussian(cfg.n, cfg.d, cfg.seed_data);
    let queries = make_gaussian(cfg.q, cfg.d, cfg.seed_query);

    let exact_idx = ExactL2::from_vectors(&base).unwrap();

    // Ground truth for recall.
    let truths: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            exact_top_k(&exact_idx, q, cfg.k)
                .into_iter()
                .map(|(i, _)| i)
                .collect()
        })
        .collect();

    let mut results = Vec::new();

    // -------- Variant 1: exact baseline --------
    {
        let t0 = Instant::now();
        let mut hits = 0usize;
        for (q, truth) in queries.iter().zip(&truths) {
            let r: Vec<usize> = exact_top_k(&exact_idx, q, cfg.k)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            for id in truth {
                if r.contains(id) {
                    hits += 1;
                }
            }
        }
        let elapsed = t0.elapsed();
        results.push(VariantResult {
            name: "exact-fp32".to_string(),
            mean_us_per_query: elapsed.as_micros() as f64 / cfg.q as f64,
            recall_at_k: hits as f64 / (cfg.q * cfg.k) as f64,
            flops_per_estimate: exact_idx.flops_per_estimate(),
        });
    }

    // -------- Variant 2: pure JL (r=32), no rerank --------
    for r in [16usize, 32, 64] {
        let jl = JlProjector::new(&base, r, 0xA5A5).unwrap();
        let t0 = Instant::now();
        let mut hits = 0usize;
        for (q, truth) in queries.iter().zip(&truths) {
            let res: Vec<usize> = jl_top_k(&jl, q, cfg.k).into_iter().map(|(i, _)| i).collect();
            for id in truth {
                if res.contains(id) {
                    hits += 1;
                }
            }
        }
        let elapsed = t0.elapsed();
        results.push(VariantResult {
            name: format!("jl-only-r{r}"),
            mean_us_per_query: elapsed.as_micros() as f64 / cfg.q as f64,
            recall_at_k: hits as f64 / (cfg.q * cfg.k) as f64,
            flops_per_estimate: jl.flops_per_estimate(),
        });
    }

    // -------- Variant 3: FINGER (JL + rerank) --------
    for &(r, rerank) in &[
        (16usize, 200usize),
        (32, 200),
        (32, 500),
        (64, 200),
        (64, 500),
        (64, 1000),
    ] {
        let f = FingerEstimator::new(&base, r, 0.0, 0xA5A5).unwrap();
        let t0 = Instant::now();
        let mut hits = 0usize;
        for (q, truth) in queries.iter().zip(&truths) {
            let res: Vec<usize> = finger_top_k(&f, q, cfg.k, rerank)
                .into_iter()
                .map(|(i, _)| i)
                .collect();
            for id in truth {
                if res.contains(id) {
                    hits += 1;
                }
            }
        }
        let elapsed = t0.elapsed();
        results.push(VariantResult {
            name: format!("finger-r{r}-rerank{rerank}"),
            mean_us_per_query: elapsed.as_micros() as f64 / cfg.q as f64,
            recall_at_k: hits as f64 / (cfg.q * cfg.k) as f64,
            flops_per_estimate: f.flops_per_estimate(),
        });
    }

    results
}
