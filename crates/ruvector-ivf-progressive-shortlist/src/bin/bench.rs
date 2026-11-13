//! Real benchmark: synthetic Gaussian-mixture IVF dataset.
//! Measures recall@10 and per-query latency for three scanners over a
//! sweep of nprobe values. Emits CSV + a markdown table on stdout.

use ruvector_ivf_progressive_shortlist::*;
use std::time::Instant;

fn bench_variant<S: Scanner>(
    label: &str,
    s: &S,
    postings: &[Posting],
    queries: &[Vec<f32>],
    truths: &[Vec<Candidate>],
    nprobe: usize,
    k: usize,
) -> (f32, f64, u64) {
    let mut rec_sum = 0.0;
    let mut dist_calls = 0u64;
    let t0 = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let out = s.search(postings, q, nprobe, k);
        rec_sum += recall(&out, &truths[qi]);
        dist_calls += out.len() as u64; // placeholder counter
    }
    let dt_us = t0.elapsed().as_micros() as f64 / queries.len() as f64;
    let rec = rec_sum / queries.len() as f32;
    println!("{:<24}  nprobe={:>3}  recall@{:>2}={:.3}  lat={:.1}µs",
        label, nprobe, k, rec, dt_us);
    (rec, dt_us, dist_calls)
}

fn main() {
    // Dataset config — kept small enough to run in under a minute on CPU
    // but large enough that the progressive gain is observable.
    let n = 50_000;
    let dim = 64;
    let k_clusters_data = 64;
    let n_queries = 200;
    let k_topk = 10;

    println!("# ivf-progressive-shortlist bench");
    println!("n={} dim={} data_clusters={} queries={} k={}",
        n, dim, k_clusters_data, n_queries, k_topk);

    let t0 = Instant::now();
    let vecs = gen_mixture(n, dim, k_clusters_data, 4.0, 42);
    println!("gen_mixture: {:.2}s", t0.elapsed().as_secs_f32());

    // Use 256 IVF centroids — standard choice for n=50k.
    let n_ivf = 256;
    let t0 = Instant::now();
    let centroids = kmeans(&vecs, n_ivf, 10, 7);
    let postings = build_postings(&vecs, centroids);
    let avg_list = postings.iter().map(|p| p.ids.len()).sum::<usize>() as f32 / n_ivf as f32;
    println!("kmeans+build: {:.2}s  avg_list={:.1}", t0.elapsed().as_secs_f32(), avg_list);

    // Build queries + truth.
    let queries = gen_mixture(n_queries, dim, k_clusters_data, 4.0, 99);
    let t0 = Instant::now();
    let truths: Vec<Vec<Candidate>> = queries.iter().map(|q| exact_topk(&vecs, q, k_topk)).collect();
    println!("truth: {:.2}s", t0.elapsed().as_secs_f32());

    // CSV header.
    println!("\nCSV:");
    println!("scanner,nprobe,recall,latency_us");

    let exhaust = Exhaustive;
    let norm_ord = NormOrderedAbandon;
    let progressive = Progressive { m_init: 32 };

    println!();
    for &nprobe in &[4usize, 8, 16, 32, 64] {
        let (r_e, l_e, _) = bench_variant("B0-exhaustive", &exhaust, &postings, &queries, &truths, nprobe, k_topk);
        let (r_n, l_n, _) = bench_variant("B1-norm-abandon", &norm_ord, &postings, &queries, &truths, nprobe, k_topk);
        let (r_p, l_p, _) = bench_variant("B2-progressive(m_init=32)", &progressive, &postings, &queries, &truths, nprobe, k_topk);
        println!("CSV,B0-exhaustive,{},{:.4},{:.1}", nprobe, r_e, l_e);
        println!("CSV,B1-norm-abandon,{},{:.4},{:.1}", nprobe, r_n, l_n);
        println!("CSV,B2-progressive,{},{:.4},{:.1}", nprobe, r_p, l_p);
        println!();
    }

    // Progressive m_init sweep at nprobe=16.
    println!("\nProgressive m_init sweep at nprobe=16:");
    for &m in &[4usize, 8, 16, 32, 64, 128] {
        let pr = Progressive { m_init: m };
        let (r, l, _) = bench_variant(
            &format!("B2-progressive(m={:>3})", m),
            &pr, &postings, &queries, &truths, 16, k_topk);
        println!("CSV,B2-progressive-m{},16,{:.4},{:.1}", m, r, l);
    }

    // Harder regime: overlapping data where tails dominate (small sep,
    // few data clusters relative to IVF cells).
    println!("\n== Harder regime: overlapping Gaussians (sep=1.0, 16 data clusters) ==");
    let vecs2 = gen_mixture(n, dim, 16, 1.0, 101);
    let t0 = Instant::now();
    let centroids2 = kmeans(&vecs2, n_ivf, 10, 17);
    let postings2 = build_postings(&vecs2, centroids2);
    println!("kmeans2: {:.2}s", t0.elapsed().as_secs_f32());
    let queries2 = gen_mixture(n_queries, dim, 16, 1.0, 303);
    let truths2: Vec<Vec<Candidate>> = queries2.iter()
        .map(|q| exact_topk(&vecs2, q, k_topk)).collect();

    for &nprobe in &[8usize, 16, 32] {
        let (r_e, l_e, _) = bench_variant("H-B0-exhaustive", &exhaust, &postings2, &queries2, &truths2, nprobe, k_topk);
        let (r_n, l_n, _) = bench_variant("H-B1-norm-abandon", &norm_ord, &postings2, &queries2, &truths2, nprobe, k_topk);
        let (r_p, l_p, _) = bench_variant("H-B2-progressive(m=32)", &progressive, &postings2, &queries2, &truths2, nprobe, k_topk);
        println!("CSV,H-B0-exhaustive,{},{:.4},{:.1}", nprobe, r_e, l_e);
        println!("CSV,H-B1-norm-abandon,{},{:.4},{:.1}", nprobe, r_n, l_n);
        println!("CSV,H-B2-progressive,{},{:.4},{:.1}", nprobe, r_p, l_p);
        println!();
    }
}
