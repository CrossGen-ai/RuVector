//! End-to-end recall + throughput demo for rotation-ensemble RaBitQ.
//!
//! For each (n, d) we build:
//!
//! * `FlatF32Index` — exact IP, our recall oracle
//! * `RabitqSingle` — K=1 baseline
//! * `RabitqEnsemble` — K ∈ {2, 4, 8}
//!
//! and report recall@10 vs flat, latency per query, and aux bytes per row.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_rabitq_ensemble::{
    FlatF32Index, Index, RabitqEnsemble, RabitqSingle, RotationKind, SearchResult,
};

fn gen_rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for _ in 0..n {
        let mut v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0_f32..=1.0)).collect();
        // L2-normalise to put IP ≡ cosine; makes recall sensitive to
        // angular quantisation error, which is what RaBitQ encodes.
        let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if nrm > 0.0 {
            let inv = nrm.recip();
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
        rows.push(v);
    }
    rows
}

fn recall_at(a: &[SearchResult], b: &[SearchResult]) -> f32 {
    let gold: std::collections::HashSet<u32> = a.iter().map(|r| r.id).collect();
    let hits = b.iter().filter(|r| gold.contains(&r.id)).count();
    hits as f32 / a.len() as f32
}

fn report<I: Index>(name: &str, idx: &I, queries: &[Vec<f32>], gold: &[Vec<SearchResult>], k: usize) {
    let t0 = Instant::now();
    let mut total_recall = 0.0_f32;
    let mut results: Vec<Vec<SearchResult>> = Vec::with_capacity(queries.len());
    for q in queries {
        results.push(idx.search(q, k));
    }
    let elapsed = t0.elapsed();
    for (res, g) in results.iter().zip(gold.iter()) {
        total_recall += recall_at(g, res);
    }
    let avg_recall = total_recall / queries.len() as f32;
    let us_per_query = elapsed.as_micros() as f32 / queries.len() as f32;
    let per_row = idx.aux_bytes() as f32 / idx.len() as f32;
    println!(
        "  {:<24}  recall@{k}={:.3}  {:>7.1}µs/q  aux={:>7.1} B/row",
        name, avg_recall, us_per_query, per_row
    );
}

fn sweep(n: usize, d: usize, nq: usize, seed: u64) {
    let k = 10;
    println!("\n== n={n:>6}  d={d:>3}  nq={nq}  k={k} ==");
    let rows = gen_rows(n, d, seed);
    let queries = gen_rows(nq, d, seed ^ 0xABCDEF);
    let flat = FlatF32Index::build(&rows);
    // Gold top-k from the exact flat index.
    let gold: Vec<Vec<SearchResult>> = queries.iter().map(|q| flat.search(q, k)).collect();

    // Latency of the exact oracle for context (not fair-matched storage).
    report("FlatF32 (oracle)", &flat, &queries, &gold, k);

    for kind in [
        RotationKind::HadamardSigned,
        RotationKind::HadamardSingle,
    ] {
        let tag = match kind {
            RotationKind::HaarDense => "haar",
            RotationKind::HadamardSigned => "hadamardSigned",
            RotationKind::HadamardSingle => "hadamardSingle",
        };
        let single = RabitqSingle::build(&rows, kind, seed ^ 0x11);
        report(&format!("single/{tag}         "), &single, &queries, &gold, k);
        for &k_rot in &[2usize, 4, 8] {
            let ens = RabitqEnsemble::build(&rows, k_rot, kind, seed ^ 0x11);
            report(
                &format!("ensemble K={k_rot}/{tag}"),
                &ens,
                &queries,
                &gold,
                k,
            );
        }
    }
    // Haar is O(D²): only run it at d <= 64 so the demo stays snappy.
    if d <= 64 {
        let kind = RotationKind::HaarDense;
        let single = RabitqSingle::build(&rows, kind, seed ^ 0x11);
        report(
            &format!("single/haar          "),
            &single,
            &queries,
            &gold,
            k,
        );
        for &k_rot in &[2usize, 4] {
            let ens = RabitqEnsemble::build(&rows, k_rot, kind, seed ^ 0x11);
            report(
                &format!("ensemble K={k_rot}/haar"),
                &ens,
                &queries,
                &gold,
                k,
            );
        }
    }
}

fn main() {
    println!("rotation-ensemble RaBitQ — recall & throughput sweep\n");
    sweep(1_000, 64, 100, 0xC0FFEE);
    sweep(5_000, 128, 100, 0xBADC0DE);
    sweep(50_000, 128, 50, 0xDEADBEEF);
    sweep(100_000, 128, 25, 0xFEEDFACE);
}
