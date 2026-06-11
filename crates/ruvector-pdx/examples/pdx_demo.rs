//! End-to-end demo: build all three scanners over a synthetic corpus, run the
//! same top-k query, and print throughput + ops counts.

use ruvector_pdx::{synth_corpus, Horizontal, PdxVertical, Scanner};
use std::time::Instant;

fn bench<S: Scanner>(name: &str, s: &S, q: &[f32], k: usize, runs: usize) -> (f64, u64) {
    // Warm-up.
    let _ = s.search(q, k);
    let start = Instant::now();
    let mut sink = 0u32;
    for _ in 0..runs {
        let r = s.search(q, k);
        sink = sink.wrapping_add(r[0].id);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let qps = runs as f64 / elapsed;
    let ops = s.last_ops();
    println!(
        "  {name:<28} qps={qps:>10.1}  ops/query={ops:>10}  sink={sink}"
    );
    (qps, ops)
}

fn main() {
    let configs: &[(usize, usize, usize)] = &[
        // (n, d, k)
        (10_000, 128, 10),
        (50_000, 128, 10),
        (10_000, 768, 10),
    ];
    for &(n, d, k) in configs {
        println!("\n== corpus n={n} d={d} k={k} ==");
        let rows = synth_corpus(n, d, 42);
        let q = synth_corpus(1, d, 99).pop().unwrap();
        let h = Horizontal::from_rows(&rows);
        let v_plain = PdxVertical::from_rows(&rows, false);
        let v_prune = PdxVertical::from_rows(&rows, true);
        let runs = if n >= 50_000 { 20 } else { 100 };
        let (h_qps, _) = bench("horizontal", &h, &q, k, runs);
        let (vp_qps, _) = bench("pdx-vertical", &v_plain, &q, k, runs);
        let (vpr_qps, _) = bench("pdx-vertical-pruned", &v_prune, &q, k, runs);
        println!(
            "  speedups: vert / horiz = {:.2}x   pruned / horiz = {:.2}x   pruned / vert = {:.2}x",
            vp_qps / h_qps,
            vpr_qps / h_qps,
            vpr_qps / vp_qps
        );
    }
}
