//! Benchmark for SOAR-IVF vs baseline variants.
//!
//! Runs the three IVF variants against the same synthetic Gaussian-mixture
//! dataset, sweeps `n_probe`, measures recall@10 and per-query µs, and prints
//! a summary table. Also reports build time and posting-list overhead.
//!
//! Numbers here are captured into the research document — this binary IS the
//! benchmark harness; there is no other source of numbers.

use std::time::Instant;

use ruvector_soar_ivf::dataset::{Dataset, DatasetConfig};
use ruvector_soar_ivf::ivf_common::IvfVariant;
use ruvector_soar_ivf::ivf_single::IvfSingle;
use ruvector_soar_ivf::ivf_soar::IvfSoar;
use ruvector_soar_ivf::ivf_spill_topk::IvfSpillTopK;
use ruvector_soar_ivf::metrics::{recall_at_k, Hit};
use ruvector_soar_ivf::sq_l2;

const K: usize = 10;

fn brute_force_topk(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<Hit> {
    let mut hits: Vec<Hit> = vectors
        .iter()
        .enumerate()
        .map(|(id, v)| Hit {
            id,
            dist: sq_l2(query, v),
        })
        .collect();
    hits.sort();
    hits.truncate(k);
    hits
}

#[derive(Copy, Clone)]
enum Mode {
    Exact,
    Approx,
}

fn eval(
    name: &str,
    mode: Mode,
    idx: &dyn IvfVariant,
    ds: &Dataset,
    gt: &[Vec<Hit>],
    n_probes: &[usize],
) -> Vec<Row> {
    let mut out = Vec::new();
    for &np in n_probes {
        let start = Instant::now();
        let mut recall_sum = 0.0f32;
        for (qi, q) in ds.queries.iter().enumerate() {
            let hits = match mode {
                Mode::Exact => idx.search(q, K, np),
                Mode::Approx => idx.search_approx(q, K, np),
            };
            recall_sum += recall_at_k(&hits, &gt[qi], K);
        }
        let elapsed = start.elapsed();
        let per_q_us = elapsed.as_secs_f64() * 1e6 / ds.queries.len() as f64;
        let recall = recall_sum / ds.queries.len() as f32;
        out.push(Row {
            variant: name.to_string(),
            n_probe: np,
            recall,
            per_query_us: per_q_us,
            memory_bytes: idx.memory_bytes(),
            spill_overhead: idx.spill_overhead(),
        });
    }
    out
}

struct Row {
    variant: String,
    n_probe: usize,
    recall: f32,
    per_query_us: f64,
    memory_bytes: usize,
    spill_overhead: usize,
}

fn print_table(rows: &[Row]) {
    println!();
    println!(
        "{:<20} {:>8} {:>10} {:>13} {:>14} {:>16}",
        "variant", "n_probe", "recall@10", "µs/query", "index_bytes", "spill_overhead"
    );
    println!("{}", "-".repeat(84));
    for r in rows {
        println!(
            "{:<20} {:>8} {:>10.4} {:>13.2} {:>14} {:>16}",
            r.variant,
            r.n_probe,
            r.recall,
            r.per_query_us,
            r.memory_bytes,
            r.spill_overhead
        );
    }
    println!();
}

fn main() {
    println!("SOAR-IVF benchmark");
    println!("==================");

    // Harder configuration: heavily overlapping blobs (sigma / center_sigma
    // ~ 0.6) with dims=128 and n_centroids > n_generative_clusters, so k-means
    // over-partitions ambiguous regions. This is where SOAR's residual-
    // orthogonality-amplified secondary picks pay off — under low
    // partition-boundary overlap ordinary top-k spilling already captures the
    // neighbourhood, and recall saturates at 1.0 for both variants.
    let cfg = DatasetConfig {
        n_vectors: 20_000,
        dims: 128,
        n_queries: 500,
        n_clusters: 16,
        sigma: 1.8,
        center_sigma: 3.0,
        seed: 0x50A5_0000_ABCD_1234,
    };

    let gen_start = Instant::now();
    let ds = Dataset::generate(cfg.clone());
    println!(
        "dataset  : n={} d={} q={} clusters={}   generated in {:.2?}",
        cfg.n_vectors,
        cfg.dims,
        cfg.n_queries,
        cfg.n_clusters,
        gen_start.elapsed()
    );

    // Ground truth
    let gt_start = Instant::now();
    let gt: Vec<Vec<Hit>> = ds
        .queries
        .iter()
        .map(|q| brute_force_topk(q, &ds.vectors, K))
        .collect();
    println!(
        "gt (brute-force top-{}) computed in {:.2?}",
        K,
        gt_start.elapsed()
    );

    // Index build. Same n_centroids for all variants (256 → ~78 vecs/list avg,
    // heavy over-partitioning of the 16 generative clusters — this is where
    // partition-boundary cuts hurt most and SOAR's orthogonality-amplified
    // secondaries have the most room to help).
    let n_centroids = 256usize;
    let iters = 10;
    let seed = 0xABCD_ABCD;

    let bs = Instant::now();
    let single = IvfSingle::build(&ds.vectors, n_centroids, iters, seed);
    println!("built IVF-Single      in {:.2?}", bs.elapsed());

    let bt2 = Instant::now();
    let topk2 = IvfSpillTopK::build(&ds.vectors, n_centroids, 2, iters, seed);
    println!("built IVF-SpillTopK/2 in {:.2?}", bt2.elapsed());

    let bt3 = Instant::now();
    let topk3 = IvfSpillTopK::build(&ds.vectors, n_centroids, 3, iters, seed);
    println!("built IVF-SpillTopK/3 in {:.2?}", bt3.elapsed());

    let bo2 = Instant::now();
    let soar2 = IvfSoar::build(&ds.vectors, n_centroids, 2, 1.0, iters, seed);
    println!("built IVF-SOAR/2 λ=1  in {:.2?}", bo2.elapsed());

    let bo3 = Instant::now();
    let soar3 = IvfSoar::build(&ds.vectors, n_centroids, 3, 1.0, iters, seed);
    println!("built IVF-SOAR/3 λ=1  in {:.2?}", bo3.elapsed());

    // Sweep n_probe.
    let n_probes = vec![1usize, 2, 4, 8, 16, 32, 64];

    println!("\n=== EXACT scoring (rescan corpus with L2²) ===");
    let mut all: Vec<Row> = Vec::new();
    all.extend(eval("IVF-Single", Mode::Exact, &single, &ds, &gt, &n_probes));
    all.extend(eval("IVF-SpillTopK/2", Mode::Exact, &topk2, &ds, &gt, &n_probes));
    all.extend(eval("IVF-SpillTopK/3", Mode::Exact, &topk3, &ds, &gt, &n_probes));
    all.extend(eval("IVF-SOAR/2 λ=1", Mode::Exact, &soar2, &ds, &gt, &n_probes));
    all.extend(eval("IVF-SOAR/3 λ=1", Mode::Exact, &soar3, &ds, &gt, &n_probes));
    print_table(&all);

    println!("=== APPROX scoring (centroid-anchored bound, NO corpus rescan) ===");
    let mut approx: Vec<Row> = Vec::new();
    approx.extend(eval("IVF-Single", Mode::Approx, &single, &ds, &gt, &n_probes));
    approx.extend(eval("IVF-SpillTopK/2", Mode::Approx, &topk2, &ds, &gt, &n_probes));
    approx.extend(eval("IVF-SpillTopK/3", Mode::Approx, &topk3, &ds, &gt, &n_probes));
    approx.extend(eval("IVF-SOAR/2 λ=1", Mode::Approx, &soar2, &ds, &gt, &n_probes));
    approx.extend(eval("IVF-SOAR/3 λ=1", Mode::Approx, &soar3, &ds, &gt, &n_probes));
    print_table(&approx);

    // Headline: SOAR vs SpillTopK at equal probe budget and equal spill,
    // in both scoring modes. Positive Δ = SOAR wins.
    for (mode_name, rows) in [("EXACT", &all), ("APPROX", &approx)] {
        for spill in [2usize, 3] {
            println!(
                "\n[{}] SOAR headline (spill={}, equal probe budget):",
                mode_name, spill
            );
            let sp = format!("IVF-SpillTopK/{}", spill);
            let so = format!("IVF-SOAR/{} λ=1", spill);
            for &np in &n_probes {
                let s = rows.iter().find(|r| r.variant == sp && r.n_probe == np).unwrap();
                let o = rows.iter().find(|r| r.variant == so && r.n_probe == np).unwrap();
                let d = o.recall - s.recall;
                println!(
                    "  n_probe={:2}  SpillTopK={:.4}   SOAR={:.4}   Δ={:+.4}",
                    np, s.recall, o.recall, d
                );
            }
        }
    }
}
