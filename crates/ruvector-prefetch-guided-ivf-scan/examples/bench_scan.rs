//! Real-data benchmark for prefetch-guided IVF scan.
//!
//! Emits CSV to stdout:
//!   dim,n_vectors,variant,lookahead,queries,avg_ns_per_query,ns_per_dot
//!
//! and a human-readable summary at the end. Uses a deterministic xorshift PRNG
//! so runs are reproducible without extra deps.

use std::hint::black_box;
use std::time::Instant;

use ruvector_prefetch_guided_ivf_scan::{
    adaptive_lookahead, scan_adaptive_prefetch, scan_fixed_prefetch, scan_no_prefetch,
    scan_strided_no_prefetch, scan_strided_prefetch, PostingList,
};

// Tiny xorshift64 — good enough for random-looking f32s. Not cryptographic.
struct Xor64(u64);
impl Xor64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        // Uniform in [-1, 1).
        let bits = (self.next_u64() >> 40) as u32; // 24 random bits
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }
}

fn make_list(dim: usize, n: usize, seed: u64) -> PostingList {
    let mut rng = Xor64::new(seed);
    let mut v = Vec::with_capacity(dim * n);
    for _ in 0..(dim * n) {
        v.push(rng.next_f32());
    }
    PostingList::new(dim, v)
}

fn make_query(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xor64::new(seed);
    (0..dim).map(|_| rng.next_f32()).collect()
}

fn bench<F: FnMut() -> usize>(iters: usize, mut f: F) -> (u128, usize) {
    // Warmup — 10% of iters or at least 1.
    let warm = (iters / 10).max(1);
    let mut sink = 0usize;
    for _ in 0..warm {
        sink = sink.wrapping_add(f());
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        sink = sink.wrapping_add(f());
    }
    (t0.elapsed().as_nanos(), black_box(sink))
}

struct Row {
    dim: usize,
    n: usize,
    variant: &'static str,
    lookahead: usize,
    queries: usize,
    avg_ns: f64,
    ns_per_dot: f64,
}

fn run_one(dim: usize, n: usize, k: usize, iters: usize, seed: u64) -> Vec<Row> {
    let list = make_list(dim, n, seed);
    // Use several queries in rotation to avoid a completely branch-predictable
    // memory access pattern.
    let queries: Vec<Vec<f32>> = (0..8).map(|i| make_query(dim, seed ^ (i as u64))).collect();

    // Correctness cross-check on iter 0.
    let baseline = scan_no_prefetch(&list, &queries[0], k);
    let fixed = scan_fixed_prefetch(&list, &queries[0], k, 4);
    let adaptive = scan_adaptive_prefetch(&list, &queries[0], k);
    assert_eq!(baseline, fixed, "fixed prefetch disagrees with baseline");
    assert_eq!(baseline, adaptive, "adaptive prefetch disagrees with baseline");

    let mut results = Vec::new();

    // Baseline
    let mut qi = 0usize;
    let (ns, _) = bench(iters, || {
        let q = &queries[qi % queries.len()];
        qi += 1;
        let hits = scan_no_prefetch(&list, q, k);
        hits.len()
    });
    results.push(Row {
        dim,
        n,
        variant: "no_prefetch",
        lookahead: 0,
        queries: iters,
        avg_ns: ns as f64 / iters as f64,
        ns_per_dot: ns as f64 / (iters as f64 * n as f64),
    });

    // Fixed lookaheads.
    for la in [1usize, 4, 8, 16] {
        let mut qi = 0usize;
        let (ns, _) = bench(iters, || {
            let q = &queries[qi % queries.len()];
            qi += 1;
            let hits = scan_fixed_prefetch(&list, q, k, la);
            hits.len()
        });
        results.push(Row {
            dim,
            n,
            variant: "fixed",
            lookahead: la,
            queries: iters,
            avg_ns: ns as f64 / iters as f64,
            ns_per_dot: ns as f64 / (iters as f64 * n as f64),
        });
    }

    // Adaptive.
    let la = adaptive_lookahead(dim * 4, 4);
    let mut qi = 0usize;
    let (ns, _) = bench(iters, || {
        let q = &queries[qi % queries.len()];
        qi += 1;
        let hits = scan_adaptive_prefetch(&list, q, k);
        hits.len()
    });
    results.push(Row {
        dim,
        n,
        variant: "adaptive",
        lookahead: la,
        queries: iters,
        avg_ns: ns as f64 / iters as f64,
        ns_per_dot: ns as f64 / (iters as f64 * n as f64),
    });

    results
}

/// Strided-access benchmark. This is where SW prefetch actually earns its
/// keep on Apple Silicon — the HW stream prefetcher can't help.
fn run_strided(dim: usize, n: usize, k: usize, iters: usize, seed: u64) -> Vec<Row> {
    let list = make_list(dim, n, seed);
    let queries: Vec<Vec<f32>> = (0..8).map(|i| make_query(dim, seed ^ (i as u64))).collect();
    // A prime stride coprime with n keeps the walk on a single cycle.
    let stride = pick_stride(n);

    // Correctness cross-check.
    let a = scan_strided_no_prefetch(&list, &queries[0], k, stride);
    let b = scan_strided_prefetch(&list, &queries[0], k, stride, 4);
    assert_eq!(a, b, "strided prefetch disagrees with strided baseline");

    let mut out = Vec::new();

    let mut qi = 0usize;
    let (ns, _) = bench(iters, || {
        let q = &queries[qi % queries.len()];
        qi += 1;
        let hits = scan_strided_no_prefetch(&list, q, k, stride);
        hits.len()
    });
    out.push(Row {
        dim,
        n,
        variant: "strided_no_pf",
        lookahead: 0,
        queries: iters,
        avg_ns: ns as f64 / iters as f64,
        ns_per_dot: ns as f64 / (iters as f64 * n as f64),
    });

    for la in [1usize, 4, 8, 16] {
        let mut qi = 0usize;
        let (ns, _) = bench(iters, || {
            let q = &queries[qi % queries.len()];
            qi += 1;
            let hits = scan_strided_prefetch(&list, q, k, stride, la);
            hits.len()
        });
        out.push(Row {
            dim,
            n,
            variant: "strided_pf",
            lookahead: la,
            queries: iters,
            avg_ns: ns as f64 / iters as f64,
            ns_per_dot: ns as f64 / (iters as f64 * n as f64),
        });
    }
    out
}

fn pick_stride(n: usize) -> usize {
    // Pick a prime not dividing n. Cover common test sizes.
    for &p in &[1009usize, 977, 733, 521, 313, 127, 97, 61, 37, 17, 7, 3] {
        if p < n && gcd(p, n) == 1 {
            return p;
        }
    }
    1
}

fn gcd(a: usize, b: usize) -> usize {
    let (mut a, mut b) = (a, b);
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn main() {
    println!("# ruvector-prefetch-guided-ivf-scan bench");
    println!("# hardware: run `uname -a; sysctl -n machdep.cpu.brand_string` from the shell");
    println!("# CSV columns:");
    println!("dim,n_vectors,variant,lookahead,queries,avg_ns_per_query,ns_per_dot");

    // Configs sweep dim × n. Keep total wallclock under ~30s in release.
    let configs = [
        (64usize, 10_000usize, 400usize),
        (128, 10_000, 300),
        (256, 10_000, 200),
        (512, 10_000, 100),
        (128, 50_000, 100),
        (128, 200_000, 30),
    ];

    let mut all: Vec<Row> = Vec::new();
    for (dim, n, iters) in configs {
        let rows = run_one(dim, n, 10, iters, 0xC0FFEE ^ (dim as u64) ^ (n as u64));
        for r in &rows {
            println!(
                "{},{},{},{},{},{:.1},{:.3}",
                r.dim, r.n, r.variant, r.lookahead, r.queries, r.avg_ns, r.ns_per_dot
            );
        }
        all.extend(rows);
    }

    // ----- Strided sweep -----
    let strided_configs = [
        (128usize, 10_000usize, 100usize),
        (128, 50_000, 30),
        (128, 200_000, 15),
        (256, 20_000, 30),
    ];
    let mut strided_all: Vec<Row> = Vec::new();
    for (dim, n, iters) in strided_configs {
        let rows = run_strided(dim, n, 10, iters, 0xBADC_0DE ^ dim as u64 ^ n as u64);
        for r in &rows {
            println!(
                "{},{},{},{},{},{:.1},{:.3}",
                r.dim, r.n, r.variant, r.lookahead, r.queries, r.avg_ns, r.ns_per_dot
            );
        }
        strided_all.extend(rows);
    }

    // Summary: per-config speedup vs baseline.
    eprintln!("\n=== Speedup summary (higher = faster than no_prefetch) ===");
    let mut i = 0;
    while i < all.len() {
        // Baseline is always the first row of each config's block.
        let base_ns = all[i].avg_ns;
        let dim = all[i].dim;
        let n = all[i].n;
        // Baseline row + 4 fixed + 1 adaptive = 6 rows per config.
        let end = (i + 6).min(all.len());
        eprintln!(
            "\n-- dim={} n={} baseline={:.1} ns/query ({:.3} ns/dot) --",
            dim, n, base_ns, all[i].ns_per_dot
        );
        for r in &all[i + 1..end] {
            let speedup = base_ns / r.avg_ns;
            eprintln!(
                "  {:>8} la={:>2}  {:>10.1} ns/query  {:>7.3} ns/dot  speedup={:.3}x",
                r.variant, r.lookahead, r.avg_ns, r.ns_per_dot, speedup
            );
        }
        i = end;
    }

    eprintln!("\n=== Strided speedup summary ===");
    let mut i = 0;
    while i < strided_all.len() {
        let base_ns = strided_all[i].avg_ns;
        let dim = strided_all[i].dim;
        let n = strided_all[i].n;
        let end = (i + 5).min(strided_all.len());
        eprintln!(
            "\n-- dim={} n={} strided_baseline={:.1} ns/query ({:.3} ns/dot) --",
            dim, n, base_ns, strided_all[i].ns_per_dot
        );
        for r in &strided_all[i + 1..end] {
            let speedup = base_ns / r.avg_ns;
            eprintln!(
                "  {:>10} la={:>2}  {:>10.1} ns/query  {:>7.3} ns/dot  speedup={:.3}x",
                r.variant, r.lookahead, r.avg_ns, r.ns_per_dot, speedup
            );
        }
        i = end;
    }
    eprintln!();
}
