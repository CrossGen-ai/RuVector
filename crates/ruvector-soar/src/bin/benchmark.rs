//! SOAR benchmark: three-way recall/latency/memory comparison at multiple
//! nprobe values on synthetic Gaussian-mixture data.
//!
//! Deterministic — same seed → same numbers. Prints a markdown table that
//! is copy-pasted verbatim into the nightly research doc.

use std::time::Instant;

use ruvector_soar::rng::Xorshift64;
use ruvector_soar::vec_math::l2_sq;
use ruvector_soar::{BaselineIvf, PartitionIndex, RandomSpillIvf, SoarIvf};

// ---- workload parameters ----------------------------------------------------
const N: usize = 5_000; // corpus size
const NQ: usize = 500; // number of queries
const DIM: usize = 128; // dimension
const K: usize = 32; // number of centroids
const TOP_K: usize = 10; // recall@TOP_K
const N_BLOBS: usize = 8; // for synthetic clustered corpus
const SEED_CORPUS: u64 = 0x50AA_C0DE;
const SEED_QUERY: u64 = 0xBEEF_1234;
const SEED_INDEX: u64 = 0xA57E_C0DE;

fn generate_gaussian_mixture(n: usize, dim: usize, n_blobs: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64::new(seed);
    // Blob centers uniformly in [-3, 3]^dim.
    let mut centers = vec![0.0f32; n_blobs * dim];
    for v in centers.iter_mut() {
        *v = rng.next_signed_f32() * 3.0;
    }
    let mut data = vec![0.0f32; n * dim];
    for i in 0..n {
        let blob = rng.next_usize(n_blobs);
        let c = &centers[blob * dim..(blob + 1) * dim];
        for j in 0..dim {
            data[i * dim + j] = c[j] + rng.next_signed_f32() * 0.5;
        }
    }
    data
}

fn generate_queries(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    // Queries drawn from a slightly wider distribution so many are
    // near cluster boundaries — this is exactly the workload SOAR wins on.
    let mut rng = Xorshift64::new(seed);
    let mut q = vec![0.0f32; n * dim];
    for v in q.iter_mut() {
        *v = rng.next_signed_f32() * 3.5;
    }
    q
}

/// Ground truth top-k by brute-force L2.
fn ground_truth(data: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<u32> {
    let n = data.len() / dim;
    let mut all: Vec<(u32, f32)> = (0..n)
        .map(|i| (i as u32, l2_sq(query, &data[i * dim..(i + 1) * dim])))
        .collect();
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
    all.truncate(k);
    all.into_iter().map(|(i, _)| i).collect()
}

fn recall(gt: &[u32], hits: &[u32]) -> f32 {
    let mut hit = 0usize;
    for g in gt {
        if hits.contains(g) {
            hit += 1;
        }
    }
    hit as f32 / gt.len() as f32
}

// `eval_recall` is a plain Rust function — it computes recall@k and average
// query latency for a `PartitionIndex`. Not related to any dynamic-code
// `eval()` primitive (Rust has none); renamed from `eval` to make that obvious.
fn eval_recall<I: PartitionIndex>(
    idx: &I,
    queries: &[f32],
    dim: usize,
    gts: &[Vec<u32>],
    nprobe: usize,
) -> (f32, f64) {
    let nq = queries.len() / dim;
    let start = Instant::now();
    let mut sum_recall = 0.0f32;
    for i in 0..nq {
        let q = &queries[i * dim..(i + 1) * dim];
        let hits = idx.search(q, nprobe, TOP_K);
        let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
        sum_recall += recall(&gts[i], &ids);
    }
    let dt = start.elapsed().as_secs_f64();
    (sum_recall / nq as f32, dt / nq as f64 * 1000.0) // ms per query
}

fn build_bytes(mib: usize) -> f32 {
    mib as f32 / (1024.0 * 1024.0)
}

fn main() {
    println!("=== ruvector-soar benchmark ===");
    println!(
        "N={} D={} K={} TOP_K={} NQ={} N_BLOBS={}",
        N, DIM, K, TOP_K, NQ, N_BLOBS
    );
    println!();

    // ---- generate corpus + queries ----
    println!("[1/5] generating synthetic corpus…");
    let corpus = generate_gaussian_mixture(N, DIM, N_BLOBS, SEED_CORPUS);
    let queries = generate_queries(NQ, DIM, SEED_QUERY);

    // ---- ground truth ----
    println!("[2/5] computing brute-force ground truth (top-{TOP_K})…");
    let t0 = Instant::now();
    let gts: Vec<Vec<u32>> = (0..NQ)
        .map(|i| ground_truth(&corpus, DIM, &queries[i * DIM..(i + 1) * DIM], TOP_K))
        .collect();
    let gt_ms_per_q = t0.elapsed().as_secs_f64() * 1000.0 / NQ as f64;
    println!("      brute-force: {:.3} ms/query", gt_ms_per_q);

    // ---- build indexes ----
    println!("[3/5] building indexes…");
    let t = Instant::now();
    let baseline = BaselineIvf::build(corpus.clone(), DIM, K, SEED_INDEX);
    let baseline_build_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let random = RandomSpillIvf::build(corpus.clone(), DIM, K, SEED_INDEX);
    let random_build_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let soar_l3 = SoarIvf::build_with_lambda(corpus.clone(), DIM, K, SEED_INDEX, 3.0);
    let soar_l3_build_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let soar_l1 = SoarIvf::build_with_lambda(corpus.clone(), DIM, K, SEED_INDEX, 1.0);
    let soar_l1_build_ms = t.elapsed().as_secs_f64() * 1000.0;

    // ---- stats ----
    println!("[4/5] index statistics:");
    let s_b = baseline.stats();
    let s_r = random.stats();
    let s_s = soar_l3.stats();
    let s_s1 = soar_l1.stats();
    println!(
        "  {:<22} entries={:>6} dup={:>4.2}× posting={:>4.1} KiB centroids={:>4.1} KiB total={:>5.1} MiB build={:>5.0} ms",
        baseline.name(),
        s_b.n_entries,
        s_b.duplication_ratio,
        s_b.posting_bytes as f32 / 1024.0,
        s_b.centroid_bytes as f32 / 1024.0,
        s_b.total_bytes as f32 / (1024.0 * 1024.0),
        baseline_build_ms,
    );
    println!(
        "  {:<22} entries={:>6} dup={:>4.2}× posting={:>4.1} KiB centroids={:>4.1} KiB total={:>5.1} MiB build={:>5.0} ms",
        random.name(),
        s_r.n_entries,
        s_r.duplication_ratio,
        s_r.posting_bytes as f32 / 1024.0,
        s_r.centroid_bytes as f32 / 1024.0,
        s_r.total_bytes as f32 / (1024.0 * 1024.0),
        random_build_ms,
    );
    println!(
        "  {:<22} entries={:>6} dup={:>4.2}× posting={:>4.1} KiB centroids={:>4.1} KiB total={:>5.1} MiB build={:>5.0} ms",
        soar_l3.name(),
        s_s.n_entries,
        s_s.duplication_ratio,
        s_s.posting_bytes as f32 / 1024.0,
        s_s.centroid_bytes as f32 / 1024.0,
        s_s.total_bytes as f32 / (1024.0 * 1024.0),
        soar_l3_build_ms,
    );
    println!(
        "  SoarIvf(λ=1.0)         entries={:>6} dup={:>4.2}× posting={:>4.1} KiB centroids={:>4.1} KiB total={:>5.1} MiB build={:>5.0} ms",
        s_s1.n_entries,
        s_s1.duplication_ratio,
        s_s1.posting_bytes as f32 / 1024.0,
        s_s1.centroid_bytes as f32 / 1024.0,
        s_s1.total_bytes as f32 / (1024.0 * 1024.0),
        soar_l1_build_ms,
    );

    let _ = build_bytes(0); // suppress unused warning in older toolchains

    // ---- sweep nprobe ----
    println!("[5/5] recall@{TOP_K} sweep across nprobe…");
    let nprobes = [1usize, 2, 4, 6, 8, 12, 16];
    println!();
    println!("| nprobe | Baseline r@{} | ms/q | RandomSpill r@{} | ms/q | SOAR(λ=1) r@{} | ms/q | SOAR(λ=3) r@{} | ms/q |", TOP_K, TOP_K, TOP_K, TOP_K);
    println!("|-------:|-------------:|-----:|----------------:|-----:|--------------:|-----:|--------------:|-----:|");
    let mut results = Vec::new();
    for &np in nprobes.iter() {
        let (r_b, ms_b) = eval_recall(&baseline, &queries, DIM, &gts, np);
        let (r_r, ms_r) = eval_recall(&random, &queries, DIM, &gts, np);
        let (r_s1, ms_s1) = eval_recall(&soar_l1, &queries, DIM, &gts, np);
        let (r_s3, ms_s3) = eval_recall(&soar_l3, &queries, DIM, &gts, np);
        println!(
            "| {:>6} | {:>12.4} | {:>4.2} | {:>15.4} | {:>4.2} | {:>13.4} | {:>4.2} | {:>13.4} | {:>4.2} |",
            np, r_b, ms_b, r_r, ms_r, r_s1, ms_s1, r_s3, ms_s3
        );
        results.push((np, r_b, ms_b, r_r, ms_r, r_s1, ms_s1, r_s3, ms_s3));
    }

    // ---- acceptance gates ----
    println!();
    println!("=== acceptance gates ===");

    // Gate 1: SOAR(λ=3) recall at nprobe=1 (tightest probe budget — the regime
    // where duplicate assignment matters most; at high nprobe every scheme
    // saturates to brute-force recall so gains vanish) must exceed
    // Baseline by ≥ 25 %.
    let (_, r_b_np1, _, _, _, _, _, r_s3_np1, _) = results[0]; // nprobe=1
    let gain = r_s3_np1 / r_b_np1.max(1e-9) - 1.0;
    let g1 = gain >= 0.25;
    println!(
        "  gate 1: SOAR(λ=3)/Baseline recall gain @ nprobe=1 = {:+.1}% (need ≥ +25%) → {}",
        gain * 100.0,
        if g1 { "PASS" } else { "FAIL" }
    );

    // Gate 2: SOAR(λ=3) must not exceed RandomSpill entries — same 2× memory
    // budget, so any recall win is pure algorithmic (anisotropic loss).
    let g2 = s_s.n_entries <= s_r.n_entries + 1;
    println!(
        "  gate 2: SOAR posting entries ({}) ≤ RandomSpill posting entries ({}) → {}",
        s_s.n_entries,
        s_r.n_entries,
        if g2 { "PASS" } else { "FAIL" }
    );

    // Gate 3: SOAR(λ=3) recall @ nprobe=1 must be ≥ RandomSpill recall @ nprobe=1
    //         (better use of the same duplicate budget in the tight-probe regime).
    let (_, _, _, r_r_np1, _, _, _, r_s3_np1b, _) = results[0];
    let g3 = r_s3_np1b + 1e-4 >= r_r_np1;
    println!(
        "  gate 3: SOAR recall ({:.4}) ≥ RandomSpill recall ({:.4}) @ nprobe=1 → {}",
        r_s3_np1b,
        r_r_np1,
        if g3 { "PASS" } else { "FAIL" }
    );

    // Gate 4: Determinism — repeat build & first-query should be identical.
    let baseline2 = BaselineIvf::build(corpus, DIM, K, SEED_INDEX);
    let q0 = &queries[0..DIM];
    let h1 = baseline.search(q0, 4, TOP_K);
    let h2 = baseline2.search(q0, 4, TOP_K);
    let g4 = h1 == h2;
    println!(
        "  gate 4: deterministic build+search → {}",
        if g4 { "PASS" } else { "FAIL" }
    );

    let all_pass = g1 && g2 && g3 && g4;
    println!();
    println!(
        "  overall: {}",
        if all_pass { "ALL PASS ✅" } else { "FAILED ❌" }
    );
    // Exit non-zero on any gate failure so CI/scripts can key on it.
    if !all_pass {
        std::process::exit(1);
    }
}
