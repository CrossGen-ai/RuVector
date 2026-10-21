//! Benchmark: FixedNprobe vs RadiusET vs HybridNprobeET on synthetic IVF data.
//!
//! Prints real wall-clock numbers from the current host. Reproduce with:
//!
//!   cargo run --release -p ruvector-ivf-radius-et --bin benchmark

use ruvector_ivf_radius_et::*;
use std::time::Instant;

fn make_data(n: usize, d: usize, seed: u64, modes: usize, spread: f32) -> FlatVectors {
    let mut rng = Rng::new(seed);
    let mut fv = FlatVectors::new(d);
    let mut centers = vec![0.0_f32; modes * d];
    for i in 0..centers.len() {
        centers[i] = spread * rng.next_gauss();
    }
    let mut buf = vec![0.0_f32; d];
    for i in 0..n {
        let m = i % modes;
        for j in 0..d {
            buf[j] = centers[m * d + j] + rng.next_gauss();
        }
        fv.push(&buf);
    }
    fv
}

fn make_queries(d: usize, nq: usize, seed: u64, modes: usize, spread: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let mut centers = vec![0.0_f32; modes * d];
    for i in 0..centers.len() {
        centers[i] = spread * rng.next_gauss();
    }
    let mut q = vec![0.0_f32; nq * d];
    for i in 0..nq {
        let m = i % modes;
        for j in 0..d {
            q[i * d + j] = centers[m * d + j] + rng.next_gauss();
        }
    }
    q
}

fn bench_one(
    label: &str,
    index: &IvfIndex,
    queries: &[f32],
    d: usize,
    k: usize,
    gt: &[Vec<(f32, u32)>],
    searcher: impl Fn(&[f32]) -> (Vec<(f32, u32)>, SearchStats),
) {
    let nq = queries.len() / d;
    let mut recall_sum = 0.0_f32;
    let mut cells_sum = 0usize;
    let mut pts_sum = 0usize;
    let t0 = Instant::now();
    for i in 0..nq {
        let q = &queries[i * d..(i + 1) * d];
        let (got, stats) = searcher(q);
        recall_sum += recall_at_k(&gt[i], &got);
        cells_sum += stats.cells_visited;
        pts_sum += stats.points_scanned;
    }
    let dt = t0.elapsed();
    let qps = nq as f64 / dt.as_secs_f64();
    let r = recall_sum / nq as f32;
    println!(
        "{label:<32} recall@{k}={:.4}  qps={:>8.0}  cells/q={:>6.2}  pts/q={:>7.1}  total={:?}",
        r,
        qps,
        cells_sum as f32 / nq as f32,
        pts_sum as f32 / nq as f32,
        dt
    );
}

fn run(name: &str, n: usize, d: usize, nlist: usize, nq: usize, k: usize, modes: usize) {
    println!("\n=== {name} ===");
    println!("n={n} d={d} nlist={nlist} nq={nq} k={k} modes={modes}");
    let vectors = make_data(n, d, 42, modes, 5.0);
    let queries = make_queries(d, nq, 7, modes, 5.0);
    let t_train = Instant::now();
    let index = IvfIndex::train(vectors.clone(), nlist, 12, 1);
    println!("train: {:?}", t_train.elapsed());

    // Report cell radius distribution.
    let mut radii: Vec<f32> = index.cell_radii.clone();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "cell radii  min={:.3}  p50={:.3}  p90={:.3}  max={:.3}",
        radii[0],
        radii[radii.len() / 2],
        radii[(radii.len() * 9) / 10],
        *radii.last().unwrap()
    );

    // Ground truth.
    let t_gt = Instant::now();
    let mut gt = Vec::with_capacity(nq);
    for i in 0..nq {
        let q = &queries[i * d..(i + 1) * d];
        gt.push(brute_force_topk(&vectors, q, k));
    }
    println!("ground-truth: {:?}", t_gt.elapsed());

    // Baselines across a few nprobe settings.
    for np in [4usize, 8, 16, 32].iter().copied().filter(|&np| np <= nlist) {
        bench_one(
            &format!("FixedNprobe np={np}"),
            &index,
            &queries,
            d,
            k,
            &gt,
            |q| FixedNprobe { nprobe: np }.search(&index, q, k),
        );
    }

    // Pure early termination (no nprobe cap).
    bench_one(
        "RadiusET (no cap)",
        &index,
        &queries,
        d,
        k,
        &gt,
        |q| RadiusET.search(&index, q, k),
    );

    // Hybrid caps.
    for cap in [8usize, 16, 32, 64].iter().copied().filter(|&c| c <= nlist) {
        bench_one(
            &format!("HybridET max={cap}"),
            &index,
            &queries,
            d,
            k,
            &gt,
            |q| HybridNprobeET { max_nprobe: cap }.search(&index, q, k),
        );
    }

    // Full scan for reference.
    bench_one(
        &format!("FixedNprobe np={nlist} (full)"),
        &index,
        &queries,
        d,
        k,
        &gt,
        |q| FixedNprobe { nprobe: nlist }.search(&index, q, k),
    );
}

fn main() {
    println!("ruvector-ivf-radius-et: real benchmark (release build)\n");
    // Compact dataset for a fast confidence pass.
    run("A: small (D=32, N=20k, nlist=128)", 20_000, 32, 128, 500, 10, 32);
    // Moderate dataset: closer to realistic IVF workloads.
    run("B: medium (D=64, N=50k, nlist=256)", 50_000, 64, 256, 500, 10, 64);
    // Harder: fewer modes than nlist, so cells are large/overlapping.
    run("C: overlap (D=64, N=50k, nlist=256, modes=16)", 50_000, 64, 256, 500, 10, 16);
}
