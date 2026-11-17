//! Real benchmark: builds Flat-IVF, IMI-small, IMI-large on identical
//! clustered data; sweeps probe/shortlist to report recall@10 vs
//! per-query latency and candidates scanned. No mocks, no stubs.

use std::time::Instant;

use ruvector_imi::data::{make_clustered, make_queries};
use ruvector_imi::kmeans::l2_sq;
use ruvector_imi::{AnnIndex, FlatIvf, ImiIndex};

fn exact_top_k(data: &[f32], d: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / d;
    let mut pairs: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| (i, l2_sq(query, &data[i as usize * d..(i as usize + 1) * d])))
        .collect();
    pairs.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    pairs.truncate(k);
    pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    pairs.into_iter().map(|(i, _)| i).collect()
}

fn recall_at_k(got: &[(u32, f32)], truth: &[u32]) -> f32 {
    let mut hit = 0usize;
    for &t in truth {
        if got.iter().any(|(g, _)| *g == t) {
            hit += 1;
        }
    }
    hit as f32 / truth.len() as f32
}

fn main() {
    let n = 100_000usize;
    let d = 128usize;
    let n_clusters = 1024usize;
    let jitter = 0.08f32;
    let n_queries = 500usize;
    let k_top = 10usize;

    println!("# ruvector-imi bench — n={n}, d={d}, n_clusters={n_clusters}, queries={n_queries}, k={k_top}");
    let data = make_clustered(n, d, n_clusters, jitter, 0xC0FFEE);
    let queries = make_queries(n_queries, d, n_clusters, jitter * 1.1, 0xBEEF);

    // Ground truth.
    println!("computing exact top-{k_top}…");
    let t0 = Instant::now();
    let truth: Vec<Vec<u32>> = (0..n_queries)
        .map(|q| exact_top_k(&data, d, &queries[q * d..(q + 1) * d], k_top))
        .collect();
    let gt_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("exact gt: {gt_ms:.1} ms total ({:.3} ms/q)", gt_ms / n_queries as f64);

    // Variant A: Flat IVF, K = sqrt(N) ≈ 316.
    let k_flat = 316usize;
    let t = Instant::now();
    let flat = FlatIvf::build(data.clone(), d, k_flat, 12, 0xA11CE);
    println!(
        "\n[A] FlatIVF   K={k_flat:>4}  build={:>7.2}s  mem={:>8.2} MiB",
        t.elapsed().as_secs_f64(),
        flat.memory_bytes() as f64 / 1024.0 / 1024.0
    );
    for &nprobe in &[2, 4, 8, 16, 32, 64] {
        run_sweep(&flat, &queries, &truth, d, n_queries, k_top, nprobe, 0);
    }

    // Variant B: IMI with K=32 per half ⇒ 1024 virtual cells.
    let k_imi = 32usize;
    let t = Instant::now();
    let mut imi = ImiIndex::build(data.clone(), d, k_imi, 20, 0xB0B);
    println!(
        "\n[B] IMI-small K={:>4}² build={:>7.2}s  mem={:>8.2} MiB",
        k_imi,
        t.elapsed().as_secs_f64(),
        imi.memory_bytes() as f64 / 1024.0 / 1024.0
    );
    for &sl in &[256usize, 512, 1024, 2048, 4096, 8192] {
        imi.shortlist_cap = sl;
        run_sweep(&imi, &queries, &truth, d, n_queries, k_top, 1 << 20, sl);
    }

    // Variant C: IMI with K=64 per half ⇒ 4096 virtual cells — the
    // headline configuration: on 100k points, cells average ~24 pts.
    let k_imi2 = 64usize;
    let t = Instant::now();
    let mut imi2 = ImiIndex::build(data.clone(), d, k_imi2, 20, 0xCAFE);
    println!(
        "\n[C] IMI-large K={:>4}² build={:>7.2}s  mem={:>8.2} MiB",
        k_imi2,
        t.elapsed().as_secs_f64(),
        imi2.memory_bytes() as f64 / 1024.0 / 1024.0
    );
    for &sl in &[256usize, 512, 1024, 2048, 4096, 8192] {
        imi2.shortlist_cap = sl;
        run_sweep(&imi2, &queries, &truth, d, n_queries, k_top, 1 << 20, sl);
    }
}

fn run_sweep(
    idx: &dyn AnnIndex,
    queries: &[f32],
    truth: &[Vec<u32>],
    d: usize,
    n_queries: usize,
    k_top: usize,
    probe: usize,
    sl: usize,
) {
    let t = Instant::now();
    let mut recall_sum = 0.0f32;
    for q in 0..n_queries {
        let got = idx.search(&queries[q * d..(q + 1) * d], probe, k_top);
        recall_sum += recall_at_k(&got, &truth[q]);
    }
    let total_ms = t.elapsed().as_secs_f64() * 1e3;
    let qps = n_queries as f64 / (total_ms / 1e3);
    let recall = recall_sum / n_queries as f32;
    let tag = if idx.name() == "flat-ivf" {
        format!("nprobe={probe:>3}")
    } else {
        format!("shortlist={sl:>5}")
    };
    println!(
        "  {} {:<16} recall@{k_top}={:.3}  lat={:>6.2} ms/q  qps={:>8.0}",
        idx.name(),
        tag,
        recall,
        total_ms / n_queries as f64,
        qps,
    );
}
