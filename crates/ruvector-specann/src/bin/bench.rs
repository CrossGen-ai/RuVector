//! specann-bench — real cargo-run benchmark producing throughput + recall numbers.
//!
//! Usage: `cargo run --release -p ruvector-specann --bin specann-bench`
//!
//! Emits JSON to stdout AND a human summary to stderr. The nightly research
//! doc pulls the JSON into the results table.

use std::time::Instant;

use rand::prelude::*;
use rand_distr::StandardNormal;
use ruvector_specann::{
    recall_at_k, DraftIndex, EscalationPolicy, F32BruteForce, Int8BruteForce, Neighbor,
    Sign1BitDraft, SpecAnnIndex,
};

fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.sample(StandardNormal)).collect())
        .collect()
}

#[derive(serde::Serialize)]
struct VariantResult {
    name: &'static str,
    recall_at_10: f64,
    qps: f64,
    p50_us: f64,
    p95_us: f64,
    avg_verified: f64,
}

fn percentile(sorted: &[u128], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx] as f64 / 1000.0 // ns -> us
}

fn bench<F>(name: &'static str, n_queries: usize, mut f: F) -> (f64, f64, f64)
where
    F: FnMut(usize) -> Vec<Neighbor>,
{
    // warmup
    for i in 0..5 {
        let _ = f(i);
    }
    let mut lats = Vec::with_capacity(n_queries);
    let t0 = Instant::now();
    for i in 0..n_queries {
        let ts = Instant::now();
        let _r = f(i);
        lats.push(ts.elapsed().as_nanos());
    }
    let secs = t0.elapsed().as_secs_f64();
    lats.sort();
    let qps = n_queries as f64 / secs;
    let p50 = percentile(&lats, 0.50);
    let p95 = percentile(&lats, 0.95);
    eprintln!(
        "  {:<28} qps={:>9.1}  p50={:>6.1}us  p95={:>7.1}us",
        name, qps, p50, p95
    );
    (qps, p50, p95)
}

fn main() {
    let n = 10_000;
    let dim = 128;
    let n_queries = 200;
    let k = 10;

    eprintln!(
        "== specann-bench == n={} dim={} n_queries={} k={}",
        n, dim, n_queries, k
    );

    let vecs = synth(n, dim, 42);
    let queries = synth(n_queries, dim, 7);

    let baseline = F32BruteForce::from_vectors(&vecs).unwrap();
    let truths: Vec<Vec<Neighbor>> = queries
        .iter()
        .map(|q| {
            let mut r = DraftIndex::draft(&baseline, q, k).unwrap();
            r.sort();
            r
        })
        .collect();

    let mut results = Vec::new();

    // ---- A: exact float32 brute force (verify-all baseline) ----
    {
        let (qps, p50, p95) = bench("A exact-f32-bruteforce", n_queries, |i| {
            DraftIndex::draft(&baseline, &queries[i], k).unwrap()
        });
        let mut rec = 0.0;
        for (q, t) in queries.iter().zip(&truths) {
            let r = DraftIndex::draft(&baseline, q, k).unwrap();
            rec += recall_at_k(&r, t, k) as f64;
        }
        results.push(VariantResult {
            name: "A_exact_f32",
            recall_at_10: rec / n_queries as f64,
            qps,
            p50_us: p50,
            p95_us: p95,
            avg_verified: n as f64,
        });
    }

    // ---- B: int8 draft only, no verification ----
    let int8 = Int8BruteForce::from_vectors(&vecs).unwrap();
    {
        let (qps, p50, p95) = bench("B int8-only-no-verify", n_queries, |i| {
            int8.draft(&queries[i], k).unwrap()
        });
        let mut rec = 0.0;
        for (q, t) in queries.iter().zip(&truths) {
            let r = int8.draft(q, k).unwrap();
            rec += recall_at_k(&r, t, k) as f64;
        }
        results.push(VariantResult {
            name: "B_int8_only",
            recall_at_10: rec / n_queries as f64,
            qps,
            p50_us: p50,
            p95_us: p95,
            avg_verified: 0.0,
        });
    }

    // ---- C: SpecANN — int8 draft + f32 verify + adaptive escalation ----
    {
        let draft = Int8BruteForce::from_vectors(&vecs).unwrap();
        let verifier = F32BruteForce::from_vectors(&vecs).unwrap();
        let spec = SpecAnnIndex::new(draft, verifier, EscalationPolicy::default());

        let mut verified_sum = 0usize;
        let (qps, p50, p95) = bench("C specann-int8+f32-verify", n_queries, |i| {
            let (r, s) = spec.search(&queries[i], k).unwrap();
            verified_sum += s.verified;
            r
        });
        let mut rec = 0.0;
        for (q, t) in queries.iter().zip(&truths) {
            let (r, _) = spec.search(q, k).unwrap();
            rec += recall_at_k(&r, t, k) as f64;
        }
        results.push(VariantResult {
            name: "C_specann_int8_f32",
            recall_at_10: rec / n_queries as f64,
            qps,
            p50_us: p50,
            p95_us: p95,
            avg_verified: verified_sum as f64 / n_queries as f64,
        });
    }

    // ---- D: SpecANN — 1-bit sign draft + f32 verify (aggressive) ----
    {
        let draft = Sign1BitDraft::from_vectors(&vecs).unwrap();
        let verifier = F32BruteForce::from_vectors(&vecs).unwrap();
        let spec = SpecAnnIndex::new(
            draft,
            verifier,
            EscalationPolicy {
                alpha: 12,
                gap_threshold: 0.05,
                max_escalations: 2,
                escalate_multiplier: 2.5,
            },
        );

        let mut verified_sum = 0usize;
        let (qps, p50, p95) = bench("D specann-1bit+f32-verify", n_queries, |i| {
            let (r, s) = spec.search(&queries[i], k).unwrap();
            verified_sum += s.verified;
            r
        });
        let mut rec = 0.0;
        for (q, t) in queries.iter().zip(&truths) {
            let (r, _) = spec.search(q, k).unwrap();
            rec += recall_at_k(&r, t, k) as f64;
        }
        results.push(VariantResult {
            name: "D_specann_1bit_f32",
            recall_at_10: rec / n_queries as f64,
            qps,
            p50_us: p50,
            p95_us: p95,
            avg_verified: verified_sum as f64 / n_queries as f64,
        });
    }

    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
