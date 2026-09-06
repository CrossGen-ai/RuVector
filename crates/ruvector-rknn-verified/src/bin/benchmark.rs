//! Benchmark: three variants.
//!
//!   1. `baseline_ann`     — NoisyAnn top-K only (simulates a coarse ANN pass).
//!   2. `rknn_live`        — baseline + live reverse-KNN verifier.
//!   3. `rknn_cached`      — baseline + pre-built reverse-KNN cache verifier.
//!
//! Metrics computed against exact top-K oracle (FlatL2Index):
//!   * recall@K
//!   * precision@K  (fraction of returned candidates in oracle set)
//!   * per-query latency (median, ns)
//!   * hub-in-result rate (fraction of returned ids that are hubs)
//!
//! No mocks. Numbers printed here go straight into the research doc.

use std::time::Instant;

use ruvector_rknn_verified::dataset::{generate, GenSpec};
use ruvector_rknn_verified::{FlatL2Index, NnIndex, NoisyAnn, RknnVerifier};

const K: usize = 10;
const BASE_M: usize = 40; // candidate list size before verification

fn median(mut xs: Vec<u128>) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

struct EvalResult {
    name: &'static str,
    recall: f64,
    precision: f64,
    hub_rate: f64,
    median_ns: u128,
    mean_returned: f64,
}

fn evaluate<F>(
    name: &'static str,
    queries: &[Vec<f32>],
    oracle: &FlatL2Index,
    hub_ids: &[usize],
    mut run: F,
) -> EvalResult
where
    F: FnMut(&[f32]) -> Vec<(usize, f32)>,
{
    let mut recalls = Vec::with_capacity(queries.len());
    let mut precisions = Vec::with_capacity(queries.len());
    let mut hub_hits = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    let mut returned_sizes = 0usize;

    for q in queries {
        let truth: std::collections::HashSet<usize> =
            oracle.search(q, K).into_iter().map(|(i, _)| i).collect();

        let t0 = Instant::now();
        let out = run(q);
        latencies.push(t0.elapsed().as_nanos());

        let out_set: std::collections::HashSet<usize> =
            out.iter().map(|(i, _)| *i).collect();
        let inter = out_set.intersection(&truth).count();
        let ret = out_set.len().max(1);
        recalls.push(inter as f64 / K as f64);
        precisions.push(inter as f64 / ret as f64);
        let hub_here = out.iter().filter(|(i, _)| hub_ids.contains(i)).count();
        hub_hits.push(hub_here as f64 / ret as f64);
        returned_sizes += ret;
    }

    EvalResult {
        name,
        recall: recalls.iter().sum::<f64>() / recalls.len() as f64,
        precision: precisions.iter().sum::<f64>() / precisions.len() as f64,
        hub_rate: hub_hits.iter().sum::<f64>() / hub_hits.len() as f64,
        median_ns: median(latencies),
        mean_returned: returned_sizes as f64 / queries.len() as f64,
    }
}

fn print_row(r: &EvalResult) {
    println!(
        "  {:<18} recall={:.3}  precision={:.3}  hub_rate={:.3}  ret_avg={:.2}  lat_med={:>8} ns",
        r.name, r.recall, r.precision, r.hub_rate, r.mean_returned, r.median_ns
    );
}

fn main() {
    let spec = GenSpec {
        n: 6_000,
        dim: 64,
        n_clusters: 24,
        hub_frac: 0.06,
        hub_scale: 3.8,
        seed: 0xA11CE,
    };
    let n_queries = 200;

    println!("== ruvector-rknn-verified benchmark ==");
    println!(
        "dataset: n={}  dim={}  clusters={}  hub_frac={:.2}  hub_scale={:.1}  queries={}",
        spec.n, spec.dim, spec.n_clusters, spec.hub_frac, spec.hub_scale, n_queries
    );
    println!("params: K={K}  base_M={BASE_M}\n");

    let build_t0 = Instant::now();
    let gen = generate(&spec, n_queries);
    let gen_ms = build_t0.elapsed().as_millis();
    println!(
        "generated dataset in {} ms ({} hubs, hub_frac_actual={:.3})",
        gen_ms,
        gen.hub_ids.len(),
        gen.hub_ids.len() as f32 / spec.n as f32
    );

    // Build noisy ANN backend (deterministic).
    let noisy = NoisyAnn::new(&gen.index, 3, 0xBEEF);

    // Pre-build cache once.
    let cache_t0 = Instant::now();
    let cached_verifier = RknnVerifier::new_cached(&gen.index, 12, 1.15);
    let cache_ms = cache_t0.elapsed().as_millis();
    let cache_bytes_est = spec.n * (12 * std::mem::size_of::<usize>() + std::mem::size_of::<f32>());
    println!(
        "built rknn cache in {} ms  (est. memory: {} KiB, {} bytes/point)\n",
        cache_ms,
        cache_bytes_est / 1024,
        cache_bytes_est / spec.n
    );

    // Live verifier (no precompute).
    let live_verifier = RknnVerifier::new_live(&gen.index, 12, 1.15);

    println!("results (means over {n_queries} queries):");

    let baseline = evaluate("baseline_ann", &gen.queries, &gen.index, &gen.hub_ids, |q| {
        noisy.search(q, BASE_M).into_iter().take(K).collect()
    });
    print_row(&baseline);

    let rknn_live = evaluate("rknn_live", &gen.queries, &gen.index, &gen.hub_ids, |q| {
        let cands = noisy.search(q, BASE_M);
        live_verifier.filter(q, &cands).into_iter().take(K).collect()
    });
    print_row(&rknn_live);

    let rknn_cached = evaluate("rknn_cached", &gen.queries, &gen.index, &gen.hub_ids, |q| {
        let cands = noisy.search(q, BASE_M);
        cached_verifier
            .filter(q, &cands)
            .into_iter()
            .take(K)
            .collect()
    });
    print_row(&rknn_cached);

    // Acceptance-test numeric checks. Fail loudly if invariants break.
    println!("\nacceptance-tests:");
    let ok_precision = rknn_cached.precision >= baseline.precision;
    let ok_hub = rknn_cached.hub_rate <= baseline.hub_rate;
    let ok_cache_faster = rknn_cached.median_ns <= rknn_live.median_ns * 4;
    println!(
        "  cached_precision >= baseline_precision  : {} ({:.3} vs {:.3})",
        ok_precision, rknn_cached.precision, baseline.precision
    );
    println!(
        "  cached_hub_rate  <= baseline_hub_rate   : {} ({:.3} vs {:.3})",
        ok_hub, rknn_cached.hub_rate, baseline.hub_rate
    );
    println!(
        "  cached_latency   <= 4x live_latency     : {} ({} vs {})",
        ok_cache_faster, rknn_cached.median_ns, rknn_live.median_ns
    );

    let all_ok = ok_precision && ok_hub && ok_cache_faster;
    if !all_ok {
        eprintln!("\nACCEPTANCE FAILED — see report");
        std::process::exit(2);
    }
    println!("\nall acceptance checks passed.");
}
