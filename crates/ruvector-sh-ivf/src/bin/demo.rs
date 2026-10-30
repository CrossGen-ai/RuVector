//! sh-ivf-demo: run the three IVF-PQ scanners on a synthetic dataset
//! and print real recall@10 and latency numbers for the research doc.

use std::time::Instant;

use ruvector_sh_ivf::{
    brute_force_topk, gen_dataset, recall, scan_full, scan_sh, scan_two_stage, IvfPq,
};

fn main() {
    // ----- parameters -----
    let d = 96;
    let n = 50_000;
    let n_data_clusters = 128;
    let n_cells = 256;
    let m_pq = 12;
    let k_pq = 256;
    let k_search = 10;
    let n_queries = 200;

    println!(
        "sh-ivf-demo: n={} d={} cells={} M={} K={} queries={}",
        n, d, n_cells, m_pq, k_pq, n_queries
    );

    // ----- build -----
    let t0 = Instant::now();
    let base = gen_dataset(n, d, n_data_clusters, 11);
    let queries = gen_dataset(n_queries, d, n_data_clusters, 22);
    println!("data generated in {:?}", t0.elapsed());

    let t0 = Instant::now();
    let idx = IvfPq::train(&base, d, n_cells, m_pq, k_pq, 7);
    println!("IVF-PQ trained in {:?}", t0.elapsed());

    // ----- ground truth -----
    let t0 = Instant::now();
    let gt: Vec<Vec<u32>> = (0..n_queries)
        .map(|i| {
            let q = &queries[i * d..(i + 1) * d];
            brute_force_topk(&base, d, q, k_search)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
        .collect();
    println!("ground truth in {:?}", t0.elapsed());

    // ----- sweep nprobe for each scanner -----
    println!();
    println!("| scanner | nprobe | recall@10 | avg latency µs/query | p95 µs |");
    println!("|---|---|---|---|---|");

    // Each row: nprobe, keep0_mult, keep1_mult, two_stage_keep0
    // These parameters sweep SH-IVF across the latency/recall Pareto front.
    // Tight settings aim at baseline recall; loose settings push latency down.
    let configs: &[(usize, usize, usize, usize)] = &[
        (8, 200, 100, 200),  // SH matches baseline recall closely
        (16, 160, 80, 160),  // SH matches baseline recall closely
        (16, 80, 30, 40),    // SH trades recall for latency
        (32, 120, 60, 120),  // SH matches baseline recall closely
        (32, 60, 25, 30),    // SH trades recall for latency
    ];

    for &(nprobe, keep0, keep1, ts_keep0) in configs {
        // Baseline full ADC
        run_bench("full", &idx, &queries, d, n_queries, nprobe, k_search, &gt, |idx, q| {
            scan_full(idx, q, nprobe, k_search)
        });

        // Two-stage: M/2 prefix then full, keep ts_keep0 * k candidates
        run_bench(
            "2-stage(M/2)",
            &idx,
            &queries,
            d,
            n_queries,
            nprobe,
            k_search,
            &gt,
            |idx, q| scan_two_stage(idx, q, nprobe, k_search, m_pq / 2, ts_keep0),
        );

        // Successive halving: M/2 (loose) → 3M/4 (tighter) → M
        run_bench(
            "SH(M/2,3M/4)",
            &idx,
            &queries,
            d,
            n_queries,
            nprobe,
            k_search,
            &gt,
            |idx, q| {
                scan_sh(
                    idx,
                    q,
                    nprobe,
                    k_search,
                    m_pq / 2,
                    (3 * m_pq) / 4,
                    keep0,
                    keep1,
                )
            },
        );
    }

    // ----- memory footprint estimate -----
    let bytes_per_vec = m_pq; // 1 byte per PQ subspace
    let posting_overhead = 4; // u32 vec_id
    let total_posting_bytes = n * (bytes_per_vec + posting_overhead);
    let centroid_bytes = n_cells * d * 4;
    let codebook_bytes = m_pq * k_pq * (d / m_pq) * 4;
    println!();
    println!("## Memory footprint (bytes)");
    println!("- centroids: {}", centroid_bytes);
    println!("- codebooks: {}", codebook_bytes);
    println!(
        "- postings:  {} ({} bytes/vec code + {} bytes id)",
        total_posting_bytes, bytes_per_vec, posting_overhead
    );
    println!(
        "- total:     {}",
        centroid_bytes + codebook_bytes + total_posting_bytes
    );
}

#[allow(clippy::too_many_arguments)]
fn run_bench<F>(
    label: &str,
    idx: &IvfPq,
    queries: &[f32],
    d: usize,
    n_queries: usize,
    nprobe: usize,
    k: usize,
    gt: &[Vec<u32>],
    f: F,
) where
    F: Fn(&IvfPq, &[f32]) -> Vec<u32>,
{
    // Warmup
    let _ = f(idx, &queries[0..d]);

    let mut recalls = 0.0f32;
    let mut times: Vec<u128> = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        let q = &queries[i * d..(i + 1) * d];
        let t = Instant::now();
        let r = f(idx, q);
        times.push(t.elapsed().as_nanos());
        let _ = k;
        recalls += recall(&gt[i], &r);
    }
    times.sort_unstable();
    let avg_ns = times.iter().sum::<u128>() as f64 / n_queries as f64;
    let p95_ns = times[(times.len() * 95) / 100] as f64;
    println!(
        "| {} | {} | {:.3} | {:.1} | {:.1} |",
        label,
        nprobe,
        recalls / n_queries as f32,
        avg_ns / 1000.0,
        p95_ns / 1000.0
    );
}
