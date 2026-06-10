//! Demo binary: builds a synthetic IVF index and prints a side-by-side
//! comparison of the three probe strategies against brute-force ground truth.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p ruvector-adaptive-probe
//! ```

use ruvector_adaptive_probe::dataset::Workload;
use ruvector_adaptive_probe::{
    FixedNprobe, IvfIndex, MarginBudget, PlateauProbe, ProbeStrategy, recall_at_k,
};
use std::time::Instant;

fn run<S: ProbeStrategy>(
    label: &str,
    idx: &IvfIndex,
    wl: &Workload,
    truth: &[Vec<(f32, u32)>],
    k: usize,
    max_nprobe: usize,
    strategy: &S,
) {
    let t0 = Instant::now();
    let mut recall_sum = 0.0f64;
    let mut probes_sum = 0usize;
    let mut scanned_sum = 0usize;
    for (qi, t) in truth.iter().enumerate().take(wl.n_queries) {
        let q = wl.query(qi);
        let (res, probes, scanned) = idx.search(q, k, max_nprobe, strategy).unwrap();
        recall_sum += recall_at_k(&res, t, k) as f64;
        probes_sum += probes;
        scanned_sum += scanned;
    }
    let dt = t0.elapsed();
    let qps = wl.n_queries as f64 / dt.as_secs_f64();
    let avg_recall = recall_sum / wl.n_queries as f64;
    let avg_probes = probes_sum as f64 / wl.n_queries as f64;
    let avg_scanned = scanned_sum as f64 / wl.n_queries as f64;
    println!(
        "{label:<22} recall@{k}={avg_recall:.4}  qps={qps:>9.1}  probes/q={avg_probes:>5.2}  pts/q={avg_scanned:>7.1}"
    );
}

fn main() {
    let n = 20_000;
    let n_queries = 500;
    let dim = 64;
    let n_clusters = 32;
    let nlist = 64;
    let k = 10;
    let max_nprobe = 16;

    println!("=== ruvector-adaptive-probe demo ===");
    println!("corpus n={n} dim={dim} clusters={n_clusters} nlist={nlist} k={k} max_nprobe={max_nprobe}");

    let wl = Workload::gaussian(n, n_queries, dim, n_clusters, 0xc0ffee);

    let t0 = Instant::now();
    let idx = IvfIndex::build(&wl.corpus, n, dim, nlist, 7).unwrap();
    println!(
        "built IVF index: n={} nlist={} mem={:.2} MB ({:.1} ms)",
        idx.n(),
        idx.nlist(),
        idx.memory_bytes() as f64 / (1024.0 * 1024.0),
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // Ground truth (brute force).
    print!("computing ground truth ... ");
    let t0 = Instant::now();
    let truth: Vec<Vec<(f32, u32)>> = (0..n_queries).map(|qi| idx.brute_force(wl.query(qi), k)).collect();
    println!("done ({:.2} s)", t0.elapsed().as_secs_f64());

    println!();
    println!("--- baselines (FixedNprobe) ---");
    for &p in &[2, 4, 8, 16] {
        run(
            &format!("fixed-nprobe={p}"),
            &idx,
            &wl,
            &truth,
            k,
            max_nprobe,
            &FixedNprobe::new(p),
        );
    }

    println!();
    println!("--- plateau (early stop on stagnant best) ---");
    for &pat in &[1usize, 2, 3] {
        run(
            &format!("plateau patience={pat}"),
            &idx,
            &wl,
            &truth,
            k,
            max_nprobe,
            &PlateauProbe::new(pat),
        );
    }

    println!();
    println!("--- margin (lower-bound prune on next centroid) ---");
    // Pick a margin that loosely matches the expected cluster radius² in this
    // dim. For sigma=0.3 and dim=64 the expected radius² is 64 * 0.3² ≈ 5.76.
    for &(m, w) in &[(0.0f32, 2usize), (3.0, 2), (6.0, 2), (12.0, 2)] {
        run(
            &format!("margin m={m:.1} w={w}"),
            &idx,
            &wl,
            &truth,
            k,
            max_nprobe,
            &MarginBudget::new(m, w),
        );
    }
}
