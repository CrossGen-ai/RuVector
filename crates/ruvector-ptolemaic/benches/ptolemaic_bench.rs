//! Standalone `harness=false` benchmark binary — runs all three backends across
//! several `(n, dim, k, num_pivots)` configurations and prints a table of real
//! numbers to stdout. No `criterion` dependency; deterministic. Invoke with
//! `cargo bench -p ruvector-ptolemaic`.

use ruvector_ptolemaic::{
    gen_dataset, gen_queries, pivot::select_farthest_first, pivot_table_bytes,
    KnnIndex, LinearScan, PtolemaicPivot, TrianglePivot,
};
use std::time::Instant;

#[derive(Copy, Clone)]
struct Config {
    n: usize,
    dim: usize,
    k: usize,
    p: usize,
    clusters: usize,
    queries: usize,
}

fn run(cfg: Config) {
    let Config {
        n,
        dim,
        k,
        p,
        clusters,
        queries: nq,
    } = cfg;
    let data = gen_dataset(n, dim, clusters, 0xC0FFEE);
    let queries = gen_queries(nq, dim, 0xBEEF);
    let pivots = select_farthest_first(&data, dim, p, 0xACE);

    let lin = LinearScan::new(data.clone(), dim).unwrap();
    let tri = TrianglePivot::build(data.clone(), dim, pivots.clone()).unwrap();
    let pto = PtolemaicPivot::build(data.clone(), dim, pivots).unwrap();

    let mut lin_dcos = 0u64;
    let mut tri_dcos = 0u64;
    let mut tri_pruned = 0u64;
    let mut pto_dcos = 0u64;
    let mut pto_pruned = 0u64;

    let t0 = Instant::now();
    for q in 0..nq {
        let qv = &queries[q * dim..(q + 1) * dim];
        let (_, s) = lin.search(qv, k).unwrap();
        lin_dcos += s.dcos;
    }
    let t_lin = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    for q in 0..nq {
        let qv = &queries[q * dim..(q + 1) * dim];
        let (_, s) = tri.search(qv, k).unwrap();
        tri_dcos += s.dcos;
        tri_pruned += s.pruned;
    }
    let t_tri = t0.elapsed().as_secs_f64();

    let t0 = Instant::now();
    for q in 0..nq {
        let qv = &queries[q * dim..(q + 1) * dim];
        let (_, s) = pto.search(qv, k).unwrap();
        pto_dcos += s.dcos;
        pto_pruned += s.pruned;
    }
    let t_pto = t0.elapsed().as_secs_f64();

    let mem_kb = pivot_table_bytes(n, p, dim) as f64 / 1024.0;
    println!(
        "n={n:>6} dim={dim:>3} k={k:>3} p={p:>2} | pivot-tbl={mem_kb:>7.1}KB | \
         Lin dcos={:>7.1}/q t={:>7.2}ms | \
         Tri dcos={:>7.1}/q ({:>5.1}% cut) t={:>7.2}ms | \
         Pto dcos={:>7.1}/q ({:>5.1}% cut) t={:>7.2}ms | \
         Pto-vs-Tri +{:>5.1}% cuts",
        lin_dcos as f64 / nq as f64,
        t_lin * 1000.0,
        tri_dcos as f64 / nq as f64,
        100.0 * (1.0 - tri_dcos as f64 / lin_dcos as f64),
        t_tri * 1000.0,
        pto_dcos as f64 / nq as f64,
        100.0 * (1.0 - pto_dcos as f64 / lin_dcos as f64),
        t_pto * 1000.0,
        100.0 * (1.0 - pto_dcos as f64 / tri_dcos.max(1) as f64),
    );
}

fn main() {
    println!("== ptolemaic_bench ==");
    let configs = [
        Config { n: 2_000, dim: 16, k: 10, p: 4, clusters: 8, queries: 50 },
        Config { n: 2_000, dim: 16, k: 10, p: 8, clusters: 8, queries: 50 },
        Config { n: 2_000, dim: 16, k: 10, p: 16, clusters: 8, queries: 50 },
        Config { n: 5_000, dim: 32, k: 10, p: 8, clusters: 12, queries: 50 },
        Config { n: 5_000, dim: 64, k: 10, p: 8, clusters: 12, queries: 50 },
        Config { n: 5_000, dim: 128, k: 10, p: 12, clusters: 12, queries: 50 },
        Config { n: 10_000, dim: 64, k: 20, p: 12, clusters: 16, queries: 50 },
        Config { n: 20_000, dim: 32, k: 10, p: 10, clusters: 20, queries: 50 },
    ];
    for cfg in configs {
        run(cfg);
    }
}
