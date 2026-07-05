//! End-to-end benchmark: build one graph, apply three layouts, run identical
//! greedy search, compare wall time + recall + edge-locality stats.
//!
//! Run: `cargo run --release -p ruvector-gorder --example bench_layouts`

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_gorder::{
    apply_permutation, layout::layout_stats, search::bench_layout, search::exact_topk, BfsLayout,
    GorderLayout, InsertionLayout, Layout, MiniHnsw, MiniHnswParams,
};

fn main() {
    // Small-but-honest benches. Keeps CI-friendly at <20s on M-series.
    let n = std::env::var("GORDER_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let dim = std::env::var("GORDER_DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let n_queries = 500usize;
    let ef = 64usize;
    let k = 10usize;

    println!("== ruvector-gorder :: cache-aware HNSW layout bench ==");
    println!("n={} dim={} ef={} k={} queries={}", n, dim, ef, k, n_queries);

    let params = MiniHnswParams { dim, m: 16, ef_construction: 64, seed: 42 };
    let t0 = std::time::Instant::now();
    let g0 = MiniHnsw::build_random(n, params.clone());
    println!("built graph in {:.2?}", t0.elapsed());

    // Query set + ground truth in original id-space.
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC0FFEE);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= nrm;
            }
            v
        })
        .collect();
    let _ = rng.gen::<u8>(); // touch rng

    print!("computing brute-force ground truth ... ");
    let t0 = std::time::Instant::now();
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| exact_topk(&g0, q, k)).collect();
    println!("done in {:.2?}", t0.elapsed());

    let layouts: Vec<(Box<dyn Layout>, &'static str)> = vec![
        (Box::new(InsertionLayout::default()), "insertion"),
        (Box::new(BfsLayout::default()), "bfs"),
        (Box::new(GorderLayout { window: 8 }), "gorder"),
    ];

    println!("\n{:<12} {:>14} {:>14} {:>10} {:>10} {:>10} {:>10}", "layout", "perm_ms", "search_QPS", "recall", "vis/q", "meanSpan", "nearFrac");
    let mut json_out = Vec::new();
    for (layout, name) in layouts {
        let t0 = std::time::Instant::now();
        let perm = layout.permutation(&g0);
        let perm_ms = t0.elapsed().as_secs_f64() * 1e3;
        let g = apply_permutation(&g0, &perm);
        let stats = bench_layout(&g, &queries, &truth, &perm, ef, k, name);
        let loc = layout_stats(&g0, &perm, name);
        println!(
            "{:<12} {:>14.2} {:>14.1} {:>10.3} {:>10.1} {:>10.1} {:>10.3}",
            name, perm_ms, stats.qps, stats.recall_at_k, stats.visited_avg, loc.mean_edge_span, loc.near_edge_frac
        );
        json_out.push(serde_json::json!({
            "layout": name,
            "perm_ms": perm_ms,
            "qps": stats.qps,
            "recall_at_k": stats.recall_at_k,
            "visited_avg": stats.visited_avg,
            "mean_edge_span": loc.mean_edge_span,
            "near_edge_frac": loc.near_edge_frac,
            "total_ns": stats.total_ns,
            "n": n, "dim": dim, "ef": ef, "k": k, "queries": n_queries,
        }));
    }
    println!("\nJSON:");
    println!("{}", serde_json::to_string_pretty(&json_out).unwrap());
}
