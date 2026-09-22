//! Real benchmark harness (no `criterion` dep to keep this crate self-contained).
//! Emits timings in nanoseconds/query so the research doc can quote them.

use ruvector_norm_bucket_ivf::{
    brute_force_topk, gaussian_dataset, mixed_scale_dataset, recall_at_k, FullScan, IvfIndex,
    NormBucketPartialL2Scan, NormBucketScan, ScanBackend, ScanStats,
};
use std::time::Instant;

#[derive(Clone, Copy)]
enum Dist {
    Gauss,
    Mixed,
}

struct Case<'a> {
    label: &'a str,
    n: usize,
    dim: usize,
    clusters: usize,
    nprobe: usize,
    dist: Dist,
}

fn measure<B: ScanBackend>(
    label: &str,
    case: &Case,
    idx: &IvfIndex,
    queries: &[f32],
    truth: &[Vec<(u32, f32)>],
    k: usize,
    backend: &B,
    repeats: usize,
) {
    let mut best_ns = u128::MAX;
    let mut agg = ScanStats::default();
    let mut recall_sum = 0f32;
    for _ in 0..repeats {
        agg = ScanStats::default();
        recall_sum = 0f32;
        let start = Instant::now();
        for (qi, chunk) in queries.chunks_exact(case.dim).enumerate() {
            let (got, s) = idx.search_with_stats(chunk, k, case.nprobe, backend);
            recall_sum += recall_at_k(&got, &truth[qi]);
            agg.full_distances += s.full_distances;
            agg.skipped_by_norm += s.skipped_by_norm;
            agg.early_abandoned += s.early_abandoned;
            agg.candidates += s.candidates;
        }
        let ns = start.elapsed().as_nanos();
        if ns < best_ns {
            best_ns = ns;
        }
    }
    let nq = queries.len() / case.dim;
    let recall = recall_sum / nq as f32;
    let ns_per_q = best_ns as f64 / nq as f64;
    let qps = 1_000_000_000.0 / ns_per_q;
    let prune_pct =
        100.0 * agg.skipped_by_norm as f64 / (agg.candidates.max(1) as f64);
    let ea_pct =
        100.0 * agg.early_abandoned as f64 / (agg.candidates.max(1) as f64);
    println!(
        "{case_lbl:<24} {label:<24} recall={recall:.3} ns/q={ns_per_q:>10.0} qps={qps:>9.1} pruned_norm={prune_pct:5.1}% early_abandon={ea_pct:5.1}%",
        case_lbl = case.label,
    );
}

fn run_case(case: &Case, nq: usize, k: usize, repeats: usize) {
    let (vectors, queries) = match case.dist {
        Dist::Gauss => (
            gaussian_dataset(case.n, case.dim, 42),
            gaussian_dataset(nq, case.dim, 1337),
        ),
        Dist::Mixed => (
            mixed_scale_dataset(case.n, case.dim, 42, 0.2, 5.0),
            mixed_scale_dataset(nq, case.dim, 1337, 0.2, 5.0),
        ),
    };
    let idx = IvfIndex::build(&vectors, case.dim, case.clusters, 15, 7);
    let mut truth = Vec::with_capacity(nq);
    for chunk in queries.chunks_exact(case.dim) {
        truth.push(brute_force_topk(&vectors, case.dim, chunk, k));
    }
    let mib = idx.estimated_bytes() as f64 / (1024.0 * 1024.0);
    println!(
        "== {} == n={} dim={} clusters={} nprobe={} k={} index={:.1} MiB",
        case.label, case.n, case.dim, case.clusters, case.nprobe, k, mib
    );
    measure(
        "full_scan",
        case,
        &idx,
        &queries,
        &truth,
        k,
        &FullScan,
        repeats,
    );
    measure(
        "norm_bucket",
        case,
        &idx,
        &queries,
        &truth,
        k,
        &NormBucketScan,
        repeats,
    );
    measure(
        "norm_bucket+partial_l2",
        case,
        &idx,
        &queries,
        &truth,
        k,
        &NormBucketPartialL2Scan,
        repeats,
    );
}

fn main() {
    let cases = [
        Case {
            label: "mixed-32d",
            n: 20_000,
            dim: 32,
            clusters: 128,
            nprobe: 8,
            dist: Dist::Mixed,
        },
        Case {
            label: "mixed-64d",
            n: 30_000,
            dim: 64,
            clusters: 256,
            nprobe: 16,
            dist: Dist::Mixed,
        },
        Case {
            label: "mixed-128d",
            n: 20_000,
            dim: 128,
            clusters: 256,
            nprobe: 16,
            dist: Dist::Mixed,
        },
        Case {
            label: "gauss-64d",
            n: 30_000,
            dim: 64,
            clusters: 256,
            nprobe: 16,
            dist: Dist::Gauss,
        },
    ];
    for c in &cases {
        run_case(c, 200, 10, 3);
    }
}
