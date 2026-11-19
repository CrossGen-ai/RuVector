//! Standalone bench harness (no criterion dependency so the workspace stays
//! light). Reports push throughput (Mops/s), finalize latency, admitted
//! fraction, and recall vs an exact top-K oracle.
//!
//! Run with `cargo bench -p ruvector-radix-bucket-topk --bench radix_bucket_bench`.

use std::time::Instant;

use ruvector_radix_bucket_topk::{
    extract::run_beam, BinaryHeapTopK, Candidate, RadixBucketTopK, SortedVecTopK, TopKCollector,
};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as u32 as f32) / (u32::MAX as f32)
}

fn make_events(n: u32, spread_exponents: f32, seed: u64) -> Vec<Candidate> {
    let mut rng = seed;
    (0..n)
        .map(|i| {
            // Log-uniform distances to stress the exponent bucketing.
            let u = lcg(&mut rng).max(1e-9);
            let d = u.powf(spread_exponents);
            Candidate::new(i, d * 10.0)
        })
        .collect()
}

fn exact_top_k(events: &[Candidate], k: usize) -> Vec<Candidate> {
    let mut v: Vec<Candidate> = events.to_vec();
    v.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    v.truncate(k);
    v
}

fn recall(got: &[Candidate], truth: &[Candidate]) -> f32 {
    let truth_ids: std::collections::HashSet<u32> = truth.iter().map(|c| c.id).collect();
    let hits = got.iter().filter(|c| truth_ids.contains(&c.id)).count();
    hits as f32 / truth.len() as f32
}

struct Row {
    name: &'static str,
    n: u32,
    k: usize,
    spread: f32,
    push_mops: f64,
    final_us: f64,
    admitted_frac: f64,
    recall: f32,
}

fn bench_one<C, F>(
    name: &'static str,
    make: F,
    events: &[Candidate],
    k: usize,
    truth: &[Candidate],
    spread: f32,
) -> Row
where
    C: TopKCollector,
    F: Fn(usize) -> C,
{
    // Warm up once.
    let _ = run_beam(make(k), events);

    let iters = 7;
    let mut best_push_ns_per_op = f64::MAX;
    let mut last_final_us = 0.0;
    let mut last_admitted = 0usize;
    let mut last_out: Vec<Candidate> = Vec::new();

    for _ in 0..iters {
        // Push-only timing: construct a collector, stream all events, do NOT
        // finalize. Admitted counting mirrors run_beam.
        let mut c = make(k);
        let mut admitted = 0usize;
        let t0 = Instant::now();
        for e in events {
            if e.dist <= c.kth_upper_bound() {
                c.push(*e);
                admitted += 1;
            }
        }
        let push_ns = t0.elapsed().as_nanos() as f64;
        let per_op = push_ns.max(1.0) / events.len() as f64;
        if per_op < best_push_ns_per_op {
            best_push_ns_per_op = per_op;
        }

        // Finalize timing: a fresh collector filled the same way, then
        // finalize measured in isolation.
        let mut c2 = make(k);
        for e in events {
            if e.dist <= c2.kth_upper_bound() {
                c2.push(*e);
            }
        }
        let tf = Instant::now();
        let out = c2.finalize();
        let fin_ns = tf.elapsed().as_nanos() as f64;
        last_final_us = fin_ns / 1000.0;
        last_admitted = admitted;
        last_out = out;
    }

    Row {
        name,
        n: events.len() as u32,
        k,
        spread,
        push_mops: 1_000.0 / best_push_ns_per_op, // ns/op -> Mops/s
        final_us: last_final_us,
        admitted_frac: last_admitted as f64 / events.len() as f64,
        recall: recall(&last_out, truth),
    }
}

fn print_header() {
    println!(
        "{:<22} {:>10} {:>5} {:>7} {:>12} {:>12} {:>10} {:>8}",
        "collector", "n", "k", "spread", "push Mops/s", "final us", "adm frac", "recall"
    );
    println!("{}", "-".repeat(90));
}

fn print_row(r: &Row) {
    println!(
        "{:<22} {:>10} {:>5} {:>7.2} {:>12.2} {:>12.2} {:>10.4} {:>8.4}",
        r.name, r.n, r.k, r.spread, r.push_mops, r.final_us, r.admitted_frac, r.recall
    );
}

fn run_regime(n: u32, k: usize, spread: f32) {
    let events = make_events(n, spread, 0xA11CEu64 ^ n as u64 ^ (k as u64));
    let truth = exact_top_k(&events, k);

    let r1 = bench_one::<BinaryHeapTopK, _>("binary-heap", |k| BinaryHeapTopK::new(k), &events, k, &truth, spread);
    let r2 = bench_one::<SortedVecTopK, _>("sorted-vec",  |k| SortedVecTopK::new(k),  &events, k, &truth, spread);
    let r3 = bench_one::<RadixBucketTopK, _>("radix-bucket", |k| RadixBucketTopK::new(k), &events, k, &truth, spread);

    print_row(&r1);
    print_row(&r2);
    print_row(&r3);
    println!();
}

fn main() {
    print_header();

    // Regime A: small beam, moderate stream (typical HNSW inner loop).
    run_regime(50_000, 10, 1.5);

    // Regime B: medium beam, large stream (ef_search ~ 200, IVF large probe).
    run_regime(500_000, 200, 1.5);

    // Regime C: narrow distance spread (all distances in [0.5, 1.5)).
    run_regime(200_000, 50, 0.1);

    // Regime D: wide spread (distances over several binary decades).
    run_regime(200_000, 50, 4.0);
}
