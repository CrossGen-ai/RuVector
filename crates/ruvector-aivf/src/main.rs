//! `aivf-demo` — runnable benchmark that measures recall@10 and per-query
//! latency for three IVF variants on a streaming, drifting workload:
//!
//! * `static` — classical IVF, centroids frozen at boot.
//! * `aivf-split` — AIVF with split-only adaptation.
//! * `aivf-split-merge` — AIVF with both split and merge.
//!
//! Numbers printed here are the source of the benchmark table in
//! docs/research/nightly/2026-06-05-aivf-adaptive-split/README.md — do not
//! edit that table by hand.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_aivf::{Aivf, AivfConfig, FlatQuantizer, l2_sq};
use std::time::Instant;

const DIM: usize = 64;
const N_BOOTSTRAP: usize = 4_000;
const N_STREAM: usize = 26_000;
const N_QUERY: usize = 1_000;
const NLIST_INIT: usize = 64;
const NPROBE: usize = 3;
const K: usize = 10;

fn gen_cluster_vec(rng: &mut StdRng, center: &[f32]) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for i in 0..DIM {
        // Gaussian-ish via 4-sum-of-uniforms.
        let g: f32 = (0..4).map(|_| rng.gen::<f32>()).sum::<f32>() - 2.0;
        v[i] = center[i] + 0.15 * g;
    }
    v
}

fn brute_force_topk(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut all: Vec<(u32, f32)> = data.iter().enumerate()
        .map(|(i, v)| (i as u32, l2_sq(q, v)))
        .collect();
    all.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    all.truncate(k);
    all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    all
}

fn recall_at_k(approx: &[(u32, f32)], truth: &[(u32, f32)]) -> f32 {
    let t: std::collections::HashSet<u32> = truth.iter().map(|x| x.0).collect();
    let hit = approx.iter().filter(|x| t.contains(&x.0)).count();
    hit as f32 / truth.len() as f32
}

struct VariantCfg { name: &'static str, splits: bool, merges: bool }

fn run_variant(cfg: &VariantCfg, all_vecs: &[Vec<f32>], queries: &[Vec<f32>], truth: &[Vec<(u32, f32)>]) {
    let mut acfg = AivfConfig::new(DIM, NLIST_INIT);
    acfg.nprobe = NPROBE;
    if !cfg.splits { acfg.split_size = usize::MAX; acfg.split_radius_factor = f32::INFINITY; }
    if !cfg.merges { acfg.merge_size = 0; }
    let q = FlatQuantizer::new(DIM);

    let bootstrap: Vec<Vec<f32>> = all_vecs[..N_BOOTSTRAP].to_vec();
    let t_build = Instant::now();
    let mut idx = Aivf::build(acfg, &bootstrap, q);
    let build_ms = t_build.elapsed().as_secs_f64() * 1000.0;

    // Streaming inserts (these are where drift happens).
    let t_stream = Instant::now();
    for i in N_BOOTSTRAP..N_BOOTSTRAP + N_STREAM {
        idx.add(i as u32, &all_vecs[i]);
    }
    let stream_ms = t_stream.elapsed().as_secs_f64() * 1000.0;

    // Queries.
    let t_q = Instant::now();
    let mut total_recall = 0.0f32;
    for (qi, q) in queries.iter().enumerate() {
        let res = idx.search(q, K);
        total_recall += recall_at_k(&res, &truth[qi]);
    }
    let q_us_per = t_q.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
    let recall = total_recall / queries.len() as f32;

    println!(
        "{:<22} | lists {:>4} | splits {:>4} | merges {:>4} | build {:>6.1}ms | stream {:>7.1}ms | search {:>6.1}µs/q | recall@{} {:.3}",
        cfg.name, idx.num_lists(), idx.split_events(), idx.merge_events(),
        build_ms, stream_ms, q_us_per, K, recall,
    );
}

fn main() {
    println!("AIVF demo — DIM={DIM} N={} nlist_init={NLIST_INIT} nprobe={NPROBE} k={K}",
             N_BOOTSTRAP + N_STREAM);

    // Deterministic RNG so benchmark numbers are reproducible.
    let mut rng = StdRng::seed_from_u64(0xA1_AB_F1_07);

    // Drifting cluster centres.  Bootstrap data uses 32 clusters; the streaming
    // tail uses 16 new clusters in *different* regions of space — this is what
    // forces the static IVF's centroids to become misaligned.
    let mut centers_boot: Vec<Vec<f32>> = (0..32).map(|_| (0..DIM).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()).collect();
    let centers_drift: Vec<Vec<f32>> = (0..16).map(|_| (0..DIM).map(|_| rng.gen::<f32>() * 6.0 - 3.0).collect()).collect();
    centers_boot.extend(centers_drift.iter().cloned());
    let centers = centers_boot;

    let mut all_vecs: Vec<Vec<f32>> = Vec::with_capacity(N_BOOTSTRAP + N_STREAM);
    for _ in 0..N_BOOTSTRAP {
        let c = &centers[rng.gen_range(0..32)];
        all_vecs.push(gen_cluster_vec(&mut rng, c));
    }
    for _ in 0..N_STREAM {
        let c = &centers[32 + rng.gen_range(0..16)];
        all_vecs.push(gen_cluster_vec(&mut rng, c));
    }

    // Queries sample mostly from the drifted region — that's the workload IVF
    // needs to keep up with.
    let mut queries: Vec<Vec<f32>> = Vec::with_capacity(N_QUERY);
    for _ in 0..N_QUERY {
        let c = &centers[32 + rng.gen_range(0..16)];
        queries.push(gen_cluster_vec(&mut rng, c));
    }

    // Ground truth via brute force (slow but exact).
    let t_truth = Instant::now();
    let truth: Vec<Vec<(u32, f32)>> = queries.iter()
        .map(|q| brute_force_topk(&all_vecs, q, K))
        .collect();
    println!("ground truth: brute force {} queries in {:.1}ms\n", N_QUERY,
             t_truth.elapsed().as_secs_f64() * 1000.0);

    println!("variant                | lists | splits | merges | build ms | stream ms | search µs/q | recall@{K}");
    println!("{}", "-".repeat(132));

    run_variant(&VariantCfg { name: "static IVF",           splits: false, merges: false }, &all_vecs, &queries, &truth);
    run_variant(&VariantCfg { name: "aivf split-only",      splits: true,  merges: false }, &all_vecs, &queries, &truth);
    run_variant(&VariantCfg { name: "aivf split+merge",     splits: true,  merges: true  }, &all_vecs, &queries, &truth);

    // Memory accounting (raw flat backing only).
    let bytes = (N_BOOTSTRAP + N_STREAM) * DIM * 4;
    println!("\nestimated raw vector memory: {:.2} MiB", bytes as f64 / (1024.0 * 1024.0));
}
