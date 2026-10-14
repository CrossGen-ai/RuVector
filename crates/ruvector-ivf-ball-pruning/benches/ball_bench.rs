//! Benchmark on gaussian-mixture synthetic data so clusters have real structure
//! and r_max is meaningfully smaller than query-centroid distances.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_ivf_ball_pruning::*;
use std::time::Instant;

fn gen_gmm(n: usize, dim: usize, modes: usize, sigma: f32, mode_scale: f32, seed: u64)
    -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Mode centers in a larger box.
    let centers: Vec<Vec<f32>> = (0..modes).map(|_|
        (0..dim).map(|_| rng.gen_range(-mode_scale..mode_scale)).collect()
    ).collect();
    // Sample points around a random mode with Gaussian noise.
    let pts: Vec<Vec<f32>> = (0..n).map(|_| {
        let m = rng.gen_range(0..modes);
        (0..dim).map(|d| {
            // Approx normal: sum of 6 uniforms minus 3 then scaled.
            let mut s = 0f32;
            for _ in 0..6 { s += rng.gen_range(0.0f32..1.0); }
            centers[m][d] + sigma * (s - 3.0)
        }).collect()
    }).collect();
    (pts, centers)
}

fn kmeans(points: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dim = points[0].len();
    let mut c: Vec<Vec<f32>> = (0..k).map(|_| points[rng.gen_range(0..points.len())].clone()).collect();
    for _ in 0..iters {
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for p in points {
            let mut best = 0; let mut bd = f32::INFINITY;
            for (ci, cc) in c.iter().enumerate() {
                let d = l2_sq(p, cc);
                if d < bd { bd = d; best = ci; }
            }
            counts[best] += 1;
            for d in 0..dim { sums[best][d] += p[d]; }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for d in 0..dim { c[ci][d] = sums[ci][d] / counts[ci] as f32; }
            }
        }
    }
    c
}

fn run(label: &str, idx: &IvfIndex, queries: &[Vec<f32>], truths: &[Vec<(u32, f32)>],
       nprobe: usize, k: usize, scan: &dyn IvfScan) {
    // warm
    for q in queries.iter().take(10) { let _ = scan.search(idx, q, nprobe, k); }
    // multiple timed trials to reduce noise, keep best
    let trials = 3;
    let mut best_elapsed = std::time::Duration::MAX;
    let mut final_agg = ScanStats::zero();
    let mut final_rec = 0f32;
    for _ in 0..trials {
        let t0 = Instant::now();
        let mut agg = ScanStats::zero();
        let mut rec_sum = 0f32;
        for (qi, q) in queries.iter().enumerate() {
            let (res, s) = scan.search(idx, q, nprobe, k);
            agg.probes_scheduled += s.probes_scheduled;
            agg.probes_actually_scanned += s.probes_actually_scanned;
            agg.vectors_touched += s.vectors_touched;
            agg.full_distances += s.full_distances;
            rec_sum += recall(&res, &truths[qi]);
        }
        let el = t0.elapsed();
        if el < best_elapsed {
            best_elapsed = el;
            final_agg = agg;
            final_rec = rec_sum / queries.len() as f32;
        }
    }
    let q_ms = best_elapsed.as_secs_f64() * 1000.0 / queries.len() as f64;
    let qps = queries.len() as f64 / best_elapsed.as_secs_f64();
    println!(
        "{:<10} nprobe={:3} rec@{}={:.4} qps={:>8.1} mean_ms={:.3} probes_scanned={:>5} touched={:>7} full_d={:>7}",
        label, nprobe, k, final_rec, qps, q_ms,
        final_agg.probes_actually_scanned / queries.len(),
        final_agg.vectors_touched / queries.len(),
        final_agg.full_distances / queries.len(),
    );
}

fn main() {
    println!("ruvector-ivf-ball-pruning :: benchmark (GMM synthetic data)");
    println!("hardware: Apple M4 Max, 16 cores");

    for &(n, dim, modes, nc, k, nq, sigma, mode_scale) in &[
        (20_000usize, 64usize, 32usize, 128usize, 10usize, 200usize, 0.25f32, 4.0f32),
        (50_000, 96, 64, 256, 10, 200, 0.25, 5.0),
        (100_000, 128, 128, 512, 10, 100, 0.25, 6.0),
    ] {
        let (data, _centers_true) = gen_gmm(n, dim, modes, sigma, mode_scale, 42);
        let (queries, _) = gen_gmm(nq, dim, modes, sigma, mode_scale, 43);
        let centroids = kmeans(&data, nc, 15, 44);
        let idx = IvfIndex::build(data, centroids);

        let t_gt = Instant::now();
        let truths: Vec<Vec<(u32, f32)>> = queries.iter()
            .map(|q| exact_topk(&idx.vectors, q, k)).collect();
        let gt_elapsed = t_gt.elapsed();

        println!();
        println!("--- n={} dim={} modes={} clusters={} k={} nq={} mean_cluster={:.1} mean_rmax={:.4} gt_time={:.2}s",
            n, dim, modes, nc, k, nq, idx.mean_cluster_size(),
            idx.r_max.iter().sum::<f32>() / idx.r_max.len() as f32,
            gt_elapsed.as_secs_f64());
        for &nprobe in &[8usize, 16, 32, 64] {
            run("baseline",   &idx, &queries, &truths, nprobe, k, &BaselineScan);
            run("partial",    &idx, &queries, &truths, nprobe, k, &PartialAbandonScan);
            run("ball-prune", &idx, &queries, &truths, nprobe, k, &BallPruneScan);
        }
    }
}
