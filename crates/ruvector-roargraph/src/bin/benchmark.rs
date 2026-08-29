//! Real benchmark harness for the three RoarGraph variants.
//!
//! Prints a table of (variant, mean_us, p50, p95, qps, memory_mb, recall@10)
//! computed against exact-search ground truth on a synthetic OOD dataset.
//! No mocks, no aspirational numbers — every field comes from `Instant::now()`
//! and `AnnIndex::memory_bytes()`.

use ruvector_roargraph::{
    cosine, gen_dataset, ground_truth, recall_at_k, AnnIndex, Entry, FlatIndex, Hit,
    KnnGraphIndex, RoarGraphIndex,
};
use std::time::Instant;

fn percentile(mut v: Vec<u128>, pct: f64) -> u128 {
    v.sort_unstable();
    if v.is_empty() {
        return 0;
    }
    let idx = ((pct / 100.0) * (v.len() as f64 - 1.0)).round() as usize;
    v[idx.min(v.len() - 1)]
}

struct Row {
    name: &'static str,
    build_ms: u128,
    mean_us: f64,
    p50_us: u128,
    p95_us: u128,
    qps: f64,
    memory_mb: f64,
    recall: f32,
}

fn bench(
    idx: &dyn AnnIndex,
    build_ms: u128,
    queries: &[Entry],
    gt: &[Vec<Hit>],
    k: usize,
) -> Row {
    let mut lat: Vec<u128> = Vec::with_capacity(queries.len());
    let mut r = 0.0f32;
    for (q, g) in queries.iter().zip(gt.iter()) {
        let t = Instant::now();
        let hits = idx.search(&q.vec, k);
        lat.push(t.elapsed().as_micros());
        r += recall_at_k(&hits, g);
    }
    let mean = lat.iter().sum::<u128>() as f64 / lat.len() as f64;
    Row {
        name: idx.name(),
        build_ms,
        mean_us: mean,
        p50_us: percentile(lat.clone(), 50.0),
        p95_us: percentile(lat, 95.0),
        qps: 1_000_000.0 / mean.max(1e-3),
        memory_mb: idx.memory_bytes() as f64 / (1024.0 * 1024.0),
        recall: r / queries.len() as f32,
    }
}

fn print_table(rows: &[Row]) {
    println!(
        "{:<18} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10} {:>10}",
        "variant", "build_ms", "mean_us", "p50_us", "p95_us", "qps", "mem_MB", "recall@10"
    );
    println!("{}", "-".repeat(96));
    for r in rows {
        println!(
            "{:<18} {:>10} {:>10.1} {:>8} {:>8} {:>10.0} {:>10.3} {:>10.3}",
            r.name, r.build_ms, r.mean_us, r.p50_us, r.p95_us, r.qps, r.memory_mb, r.recall
        );
    }
}

fn main() {
    // OOD scenario: base cloud shifted +0.6, query cloud shifted -0.6.
    let n_base = 5_000;
    let n_query_total = 400;
    let dim = 64;
    let k = 10;

    println!("=== ruvector-roargraph benchmark ===");
    println!(
        "n_base={n_base}  n_query_total={n_query_total}  dim={dim}  k={k}  (OOD: base +0.6, query -0.6)"
    );

    let (base, all_queries) = gen_dataset(n_base, n_query_total, dim, 0.6, 0.6, 20260829);
    // First half = query workload sample RoarGraph gets to see at build time.
    let split = all_queries.len() / 2;
    let workload: Vec<Entry> = all_queries[..split].to_vec();
    let eval: Vec<Entry> = all_queries[split..].to_vec();
    println!("workload={}  eval={}", workload.len(), eval.len());

    // Ground truth from exact search.
    println!("Computing exact ground truth ...");
    let t_gt = Instant::now();
    let gt = ground_truth(&base, &eval, k);
    println!("  ground_truth ready in {} ms", t_gt.elapsed().as_millis());

    // 1) Flat
    let t = Instant::now();
    let flat = FlatIndex::build(&base);
    let flat_build = t.elapsed().as_millis();

    // 2) k-NN graph
    let t = Instant::now();
    let knn = KnnGraphIndex::build(&base, 16, 48, 12, 7);
    let knn_build = t.elapsed().as_millis();

    // 3) RoarGraph-lite (same base graph budget + 8 augmentation slots).
    let t = Instant::now();
    let roar = RoarGraphIndex::build(&base, &workload, 16, 8, 8, 48, 12);
    let roar_build = t.elapsed().as_millis();
    println!(
        "RoarGraph augmentation edges added: {}",
        roar.aug_edges()
    );

    let rows = vec![
        bench(&flat, flat_build, &eval, &gt, k),
        bench(&knn, knn_build, &eval, &gt, k),
        bench(&roar, roar_build, &eval, &gt, k),
    ];
    print_table(&rows);

    // Sanity: keep the compiler / linker honest so the flat scan can't be
    // optimized away in future refactors — touch `cosine` once.
    let sink = cosine(&base[0].vec, &eval[0].vec);
    println!("sanity cosine(base[0], eval[0]) = {sink:.4}");

    // Print a compact JSON block for easy scraping by future harnesses.
    println!("\n--- machine-readable ---");
    print!("{{\"results\":[");
    for (i, r) in rows.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!(
            "{{\"name\":\"{}\",\"build_ms\":{},\"mean_us\":{:.2},\"p50_us\":{},\"p95_us\":{},\"qps\":{:.0},\"mem_mb\":{:.3},\"recall\":{:.4}}}",
            r.name, r.build_ms, r.mean_us, r.p50_us, r.p95_us, r.qps, r.memory_mb, r.recall
        );
    }
    println!("]}}");
}
