//! Benchmark binary for ruvector-ivf-hnsw-coarse.
//!
//! Emits real numbers used in the research doc. No mocks, no synthetic
//! constants. Prints a compact table across three coarse quantizers at
//! matched nprobe.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_ivf_hnsw_coarse::ivf::kmeans;
use ruvector_ivf_hnsw_coarse::{
    hnsw::HnswConfig, rproj::RandomProjConfig, sq_l2, FlatCoarse, HnswCoarse, IvfCoarse,
    RandomProjCoarse,
};

fn gen(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

/// Gaussian-mixture data: `n_clusters` clusters, each isotropic sigma=0.15,
/// centers uniform in `[-1,1]^d`. This is the standard workload where IVF
/// clustering actually pays off.
fn gen_gm(n: usize, d: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<f32> = (0..n_clusters * d)
        .map(|_| rng.gen_range(-1.0f32..1.0))
        .collect();
    let mut out = Vec::with_capacity(n * d);
    for _ in 0..n {
        let ci: usize = rng.gen_range(0..n_clusters);
        for j in 0..d {
            // Box-Muller-lite: sum of 12 uniforms - 6 ≈ N(0,1); cheap.
            let mut s = 0.0f32;
            for _ in 0..12 {
                s += rng.gen::<f32>();
            }
            s -= 6.0;
            out.push(centers[ci * d + j] + 0.15 * s);
        }
    }
    out
}

fn ground_truth(data: &[f32], queries: &[f32], d: usize, k: usize) -> Vec<Vec<u32>> {
    let n = data.len() / d;
    let nq = queries.len() / d;
    let mut gt = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &queries[qi * d..(qi + 1) * d];
        let mut scored: Vec<(f32, u32)> = (0..n)
            .map(|i| (sq_l2(q, &data[i * d..(i + 1) * d]), i as u32))
            .collect();
        scored.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        gt.push(scored.into_iter().map(|(_, i)| i).collect());
    }
    gt
}

fn run_backend<C: IvfCoarse>(
    label: &str,
    data: &[f32],
    queries: &[f32],
    centroids: &[f32],
    postings: &[Vec<u32>],
    d: usize,
    mut coarse: C,
    k: usize,
    nprobe: usize,
    gt: &[Vec<u32>],
) {
    let t0 = std::time::Instant::now();
    coarse.fit(centroids, d).expect("fit");
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let nq = queries.len() / d;
    let mut coarse_ns: u128 = 0;
    let mut scan_ns: u128 = 0;
    let mut hits = 0usize;
    let mut denom = 0usize;
    for qi in 0..nq {
        let q = &queries[qi * d..(qi + 1) * d];
        let t0 = std::time::Instant::now();
        let lists = coarse.top_lists(q, nprobe);
        let t1 = std::time::Instant::now();
        let mut out: Vec<(f32, u32)> = Vec::with_capacity(k * 4);
        for l in lists {
            for &row in &postings[l as usize] {
                let v = &data[row as usize * d..(row as usize + 1) * d];
                out.push((sq_l2(q, v), row));
            }
        }
        let take_k = k.min(out.len()).max(1);
        if !out.is_empty() {
            out.select_nth_unstable_by(take_k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            out.truncate(take_k);
        }
        let t2 = std::time::Instant::now();
        coarse_ns += (t1 - t0).as_nanos();
        scan_ns += (t2 - t1).as_nanos();
        let found: std::collections::HashSet<u32> = out.iter().map(|x| x.1).collect();
        for &g in &gt[qi] {
            if found.contains(&g) {
                hits += 1;
            }
            denom += 1;
        }
    }
    let coarse_us = coarse_ns as f64 / 1_000.0 / nq as f64;
    let scan_us = scan_ns as f64 / 1_000.0 / nq as f64;
    println!(
        "  {:22}  R@{}={:.3}  coarse={:>7.1} us  scan={:>7.1} us  total={:>7.1} us  build={:>7.1} ms  extra={:>6} KiB",
        label,
        k,
        hits as f64 / denom as f64,
        coarse_us,
        scan_us,
        coarse_us + scan_us,
        build_ms,
        coarse.extra_bytes() / 1024
    );
}

fn main() {
    println!("== ruvector-ivf-hnsw-coarse benchmark ==");
    // Scenario sized to make coarse selection a visible fraction of latency.
    // With C=4096, FlatCoarse scan alone is ~4096*d flops per query.
    let d = 64;
    let n = 50_000;
    let c = 4096;
    let nq = 300;
    let k = 10;
    let nprobe = 64;
    let n_clusters = 256;

    println!(
        "config: d={d} n={n} C={c} nq={nq} k={k} nprobe={nprobe} n_clusters={n_clusters} (Gaussian mixture, sigma=0.15)"
    );

    let data = gen_gm(n, d, n_clusters, 1);
    let queries = gen_gm(nq, d, n_clusters, 2);
    let _ = gen; // keep symbol for future workloads
    println!("computing exact ground truth ({nq} queries x {n} rows)...");
    let t0 = std::time::Instant::now();
    let gt = ground_truth(&data, &queries, d, k);
    println!("  gt done in {:.2} s", t0.elapsed().as_secs_f64());

    println!("training k-means (C={c}, 8 iters, shared across backends)...");
    let t0 = std::time::Instant::now();
    let centroids = kmeans(&data, d, c, 8, 42);
    println!("  kmeans done in {:.2} s", t0.elapsed().as_secs_f64());

    println!("assigning postings...");
    let t0 = std::time::Instant::now();
    let mut postings: Vec<Vec<u32>> = vec![Vec::new(); c];
    for i in 0..n {
        let row = &data[i * d..(i + 1) * d];
        let mut best = 0u32;
        let mut bd = f32::INFINITY;
        for cc in 0..c {
            let dd = sq_l2(row, &centroids[cc * d..(cc + 1) * d]);
            if dd < bd {
                bd = dd;
                best = cc as u32;
            }
        }
        postings[best as usize].push(i as u32);
    }
    println!("  postings done in {:.2} s", t0.elapsed().as_secs_f64());

    run_backend(
        "FlatCoarse           ",
        &data,
        &queries,
        &centroids,
        &postings,
        d,
        FlatCoarse::new(),
        k,
        nprobe,
        &gt,
    );

    for ef in [16usize, 32, 64] {
        let name = format!("HnswCoarse(ef={:>2})    ", ef);
        run_backend(
            &name,
            &data,
            &queries,
            &centroids,
            &postings,
            d,
            HnswCoarse::with_config(HnswConfig {
                ef_construction: 100,
                ef_search: ef,
                m: 16,
                m0: 32,
                seed: 7,
            }),
            k,
            nprobe,
            &gt,
        );
    }

    for bits in [64usize, 128, 256] {
        let name = format!("RandomProjCoarse({:>3})", bits);
        run_backend(
            &name,
            &data,
            &queries,
            &centroids,
            &postings,
            d,
            RandomProjCoarse::with_config(RandomProjConfig { bits, seed: 11 }),
            k,
            nprobe,
            &gt,
        );
    }
    println!("== done ==");
}
