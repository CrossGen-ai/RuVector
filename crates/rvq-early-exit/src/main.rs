//! rvq-bench — real benchmark binary for RVQ early-exit vs full-depth ADC.
//!
//! Prints a markdown-ready table suitable for inclusion in the research doc.

use rvq_early_exit::{
    gen_clustered, recall_at_k, scan_early_exit, scan_exact, scan_full, Rng, Rvq,
};
use std::time::Instant;

fn main() {
    let d = 128;
    let n = 20_000;
    let nq = 200;
    let n_clusters = 64;
    let k_codes = 256;
    let top_k = 10;

    println!("## rvq-early-exit benchmark");
    println!();
    println!(
        "dataset: synthetic clustered Gaussian, N={}, D={}, clusters={}, queries={}, top_k={}",
        n, d, n_clusters, nq, top_k
    );
    println!("codebook: K={} per stage, Lloyd iterations=12", k_codes);
    println!();

    let mut rng = Rng::new(0xC0FFEE);
    let data = gen_clustered(n, d, n_clusters, &mut rng);
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|_| {
            let mut q = vec![0.0f32; d];
            // Pick a random cluster center + noise.
            let c_idx = (rng.next_u64() as usize) % n_clusters;
            for t in 0..d {
                q[t] = data[c_idx][t] + rng.gaussian() * 0.6;
            }
            q
        })
        .collect();

    // Ground truth (exact).
    let t = Instant::now();
    let gts: Vec<Vec<_>> = queries
        .iter()
        .map(|q| scan_exact(&data, q, top_k))
        .collect();
    let exact_ms = t.elapsed().as_secs_f64() * 1000.0 / nq as f64;

    println!("| Variant | Stages M | Shortlist | Build (ms) | Avg stages decoded | Query (µs) | Recall@{} |", top_k);
    println!("|---|---|---|---|---|---|---|");

    // Variants: three M values. For each, measure full vs early-exit.
    for &m in &[4usize, 6, 8] {
        let t = Instant::now();
        let rvq = Rvq::train_and_encode(&data, m, k_codes, 12, &mut rng);
        let pnorms = rvq.build_partial_norms();
        let build_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Full-depth ADC.
        let t = Instant::now();
        let mut full_recall = 0.0f32;
        let mut full_hits = Vec::with_capacity(nq);
        for (qi, q) in queries.iter().enumerate() {
            let h = scan_full(&rvq, q, &pnorms, top_k);
            full_recall += recall_at_k(&gts[qi], &h);
            full_hits.push(h);
        }
        let full_us = t.elapsed().as_secs_f64() * 1_000_000.0 / nq as f64;
        full_recall /= nq as f32;
        println!(
            "| RVQ-full      | {} | {} | {:.1} | {:.2} | {:.1} | {:.3} |",
            m, n, build_ms, m as f32, full_us, full_recall
        );

        // Early exit: spill = 8, stable_streak = 2.
        let spill = 8usize;
        let stable_streak = 2usize;
        let t = Instant::now();
        let mut ee_recall = 0.0f32;
        let mut stages_sum = 0usize;
        let mut shortlist_total = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let (h, stats) = scan_early_exit(&rvq, q, &pnorms, top_k, spill, stable_streak);
            ee_recall += recall_at_k(&gts[qi], &h);
            stages_sum += stats.stages_used;
            shortlist_total += stats.shortlist_size;
        }
        let ee_us = t.elapsed().as_secs_f64() * 1_000_000.0 / nq as f64;
        ee_recall /= nq as f32;
        let avg_stages = stages_sum as f32 / nq as f32;
        let avg_sl = shortlist_total / nq;
        println!(
            "| RVQ-earlyexit | {} | {} | — | {:.2} | {:.1} | {:.3} |",
            m, avg_sl, avg_stages, ee_us, ee_recall
        );
    }

    println!();
    println!("exact baseline (brute force, D=128 L2): {:.1} µs/query", exact_ms * 1000.0);
    println!();
    println!("Legend: 'Shortlist' = post-stage-0 candidates refined by later stages.");
    println!("        Early exit stops when top-K is stable for 2 consecutive stages.");
}
