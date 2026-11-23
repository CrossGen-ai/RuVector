//! Benchmark binary: prints a markdown-friendly comparison across three
//! backends and multiple M (bit-plane depth) choices.
//!
//! Variants measured (>= 3 as required):
//!   1. full_sq8         — baseline, scan all 8 bits.
//!   2. top_m_approx(M)  — approximate-only (no refine), for M in {2,3,4,6}.
//!   3. progressive(M,r) — M planes shortlist, refine with full SQ8.
//!
//! Numbers reported are real cargo-run timings on the host.

use ruvector_bit_plane_sq::*;

fn fmt_us(us: u128, nq: usize) -> String {
    let per_q = us as f64 / nq as f64;
    format!("{:>9.1} us/q  ({:>7} us total)", per_q, us)
}

fn main() {
    let n: usize = std::env::var("BPP_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let dim: usize = std::env::var("BPP_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let nq: usize = std::env::var("BPP_NQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let k: usize = 10;

    eprintln!(
        "Building dataset n={} dim={} nq={} k={}",
        n, dim, nq, k
    );
    let data = synth_dataset(n, dim, 11);
    let queries = synth_dataset(nq, dim, 23);
    let sq = Sq8Codes::fit_quantize(&data, dim);
    let bp = BitPlanes::from_sq8(&sq);
    let qcodes = quantize_queries(&queries, &sq);
    let truth = gt_topk(&data, dim, &queries, k);

    let bytes_full = sq.codes.len();
    let bytes_planes_per_m: Vec<(usize, usize)> = (1..=8)
        .map(|m| {
            // Each plane is n*words u64 = n * ceil(dim/64) * 8 bytes.
            let per_plane = bp.n * bp.words * 8;
            (m, per_plane * m)
        })
        .collect();

    println!();
    println!("# BPP-SQ Bench");
    println!(
        "n={}, dim={}, nq={}, k={}, full_sq8_bytes={}",
        n, dim, nq, k, bytes_full
    );
    println!();
    println!("| backend                | recall@{} | latency                  | code bytes read |", k);
    println!("|------------------------|-----------|--------------------------|-----------------|");

    // Baseline
    let full = FullSq8 { sq: &sq };
    let (us, preds) = time_search(&qcodes, dim, |q| full.search(q, k));
    let rec = recall_at_k(&preds, &truth, k);
    println!(
        "| {:<22} | {:>9.3} | {} | {:>15} |",
        full.name(),
        rec,
        fmt_us(us, nq),
        bytes_full
    );

    for m in [2usize, 3, 4, 6] {
        let approx = bp.planes_to_approx_codes(m);
        let ta = TopMApprox {
            approx_codes: &approx,
            n: bp.n,
            dim: bp.dim,
            name_str: Box::leak(format!("top_m_approx(M={})", m).into_boxed_str()),
        };
        let (us, preds) = time_search(&qcodes, dim, |q| ta.search(q, k));
        let rec = recall_at_k(&preds, &truth, k);
        let planes_bytes = bytes_planes_per_m
            .iter()
            .find(|(mm, _)| *mm == m)
            .map(|(_, b)| *b)
            .unwrap_or(0);
        println!(
            "| {:<22} | {:>9.3} | {} | {:>15} |",
            ta.name(),
            rec,
            fmt_us(us, nq),
            planes_bytes
        );

        for r in [2usize, 4, 8] {
            let prog = Progressive {
                approx_codes: &approx,
                sq: &sq,
                top_m: m,
                shortlist_mul: r,
                name_str: Box::leak(
                    format!("progressive(M={},r={})", m, r).into_boxed_str(),
                ),
            };
            let (us, preds) = time_search(&qcodes, dim, |q| prog.search(q, k));
            let rec = recall_at_k(&preds, &truth, k);
            let refine_bytes = k * r * dim;
            println!(
                "| {:<22} | {:>9.3} | {} | {:>15} |",
                prog.name(),
                rec,
                fmt_us(us, nq),
                planes_bytes + refine_bytes
            );
        }
    }

    println!();
    println!("# Memory layout (bytes)");
    println!("- Full SQ8 codes:          {}", bytes_full);
    for (m, b) in &bytes_planes_per_m {
        println!("- Top-{} bit-planes only:   {}", m, b);
    }
}
