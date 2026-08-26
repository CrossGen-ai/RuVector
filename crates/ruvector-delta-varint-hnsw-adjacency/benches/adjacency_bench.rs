//! Real benchmark (no criterion): three variants x two locality regimes.
//!
//! Prints CSV to stdout so the research doc can capture the numbers verbatim.

use ruvector_delta_varint_hnsw_adjacency::*;
use std::time::Instant;

fn bench_decode<S: AdjacencyStore>(store: &S, iters: usize, n: usize) -> (f64, u64) {
    let m = store.max_degree();
    let mut buf = vec![0u32; m];
    let mut checksum: u64 = 0;
    let t = Instant::now();
    for it in 0..iters {
        let node = ((it * 2654435761) % n) as u32;
        let got = store.decode_into(node, &mut buf);
        // consume result to prevent DCE
        for &v in &buf[..got] {
            checksum = checksum.wrapping_add(v as u64);
        }
    }
    let ns_per_op = t.elapsed().as_nanos() as f64 / iters as f64;
    (ns_per_op, checksum)
}

fn run(n: usize, m: usize, locality: f32, label: &str) {
    let nodes = synth_neighbors(n, m, locality, 0xABCDEF01_2345_6789);
    let t0 = Instant::now();
    let plain = PlainAdjacency::build(nodes.clone(), m);
    let t_plain_build = t0.elapsed().as_micros();
    let t0 = Instant::now();
    let dv = DeltaVarintAdjacency::build(nodes.clone(), m);
    let t_dv_build = t0.elapsed().as_micros();
    let t0 = Instant::now();
    let pfor = PforBlockedAdjacency::build(nodes.clone(), m);
    let t_pfor_build = t0.elapsed().as_micros();

    let iters = 2_000_000;
    let (ns_plain, cs1) = bench_decode(&plain, iters, n);
    let (ns_dv, cs2) = bench_decode(&dv, iters, n);
    let (ns_pfor, cs3) = bench_decode(&pfor, iters, n);
    // Sanity: same checksum across all three (same graph, same access pattern).
    assert_eq!(cs1, cs2, "plain vs delta-varint checksum mismatch");
    assert_eq!(cs1, cs3, "plain vs pfor checksum mismatch");

    let fp_plain = footprint("plain-u32", &plain);
    let fp_dv = footprint("delta-varint", &dv);
    let fp_pfor = footprint("pfor-blocked", &pfor);

    println!(
        "\n== {} (n={}, M={}, locality={:.2}) ==",
        label, n, m, locality
    );
    println!("build_us: plain={} dv={} pfor={}", t_plain_build, t_dv_build, t_pfor_build);
    println!("{}", fp_plain);
    println!("{}  (x{:.2} smaller)", fp_dv, fp_plain.bytes as f64 / fp_dv.bytes as f64);
    println!("{}  (x{:.2} smaller)", fp_pfor, fp_plain.bytes as f64 / fp_pfor.bytes as f64);
    println!(
        "decode_ns_per_node: plain={:.1} dv={:.1} (x{:.2}) pfor={:.1} (x{:.2})",
        ns_plain,
        ns_dv,
        ns_dv / ns_plain,
        ns_pfor,
        ns_pfor / ns_plain,
    );
    println!(
        "CSV,{},{},{},{:.3},{},{},{},{:.2},{:.2},{:.2}",
        label, n, m, locality,
        fp_plain.bytes, fp_dv.bytes, fp_pfor.bytes,
        ns_plain, ns_dv, ns_pfor
    );
}

fn main() {
    println!("CSV,label,n,M,locality,bytes_plain,bytes_dv,bytes_pfor,ns_plain,ns_dv,ns_pfor");
    // Small — smoke.
    run(10_000, 16, 0.95, "small-local");
    run(10_000, 16, 0.00, "small-uniform");
    // Medium — headline number.
    run(100_000, 32, 0.98, "medium-local");
    run(100_000, 32, 0.50, "medium-mixed");
    run(100_000, 32, 0.00, "medium-uniform");
    // Large-ish (still fits comfortably in RAM).
    run(500_000, 32, 0.98, "large-local");
}
