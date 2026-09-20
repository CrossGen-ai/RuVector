//! Real-numbers benchmark for partial L2 with early abandonment.
//!
//! Simulates a rerank stage: N candidates, dim=768, k=10, Q queries.
//! Reports total time and speedup for full L2 vs early-abandon with
//! strides {32, 64, 128}. Also reports the fraction of vectors that
//! were abandoned (efficiency signal) and confirms recall parity.
//!
//! Run:
//!   cargo run --release -p ruvector-partial-l2 --example bench

use std::time::Instant;

use ruvector_partial_l2::{
    l2_early_abandon, l2_early_abandon_dyn, l2_full, rerank_early_abandon,
    rerank_early_abandon_dyn, rerank_full, AbandonResult, TopK,
};

fn make(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as u32 as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect()
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn measure_full(q: &[f32], vecs: &[Vec<f32>], cands: &[u32], k: usize) -> (f64, Vec<(f32, u32)>) {
    let t = Instant::now();
    let mut last = Vec::new();
    // Repeat 5 times to reduce timer noise.
    let reps = 5;
    for _ in 0..reps {
        last = rerank_full(q, vecs, cands, k);
    }
    (t.elapsed().as_secs_f64() / reps as f64, last)
}

fn measure_ea(
    q: &[f32],
    vecs: &[Vec<f32>],
    cands: &[u32],
    k: usize,
    stride: usize,
) -> (f64, Vec<(f32, u32)>, f64) {
    // Also count abandonment fraction with a separate pass so timing
    // isn't polluted.
    let mut top = TopK::new(k);
    let mut abandoned = 0usize;
    for &c in cands {
        let tau = top.threshold();
        match l2_early_abandon_dyn(q, &vecs[c as usize], stride, tau) {
            AbandonResult::Full(d) => {
                top.push(d, c);
            }
            AbandonResult::Abandoned(_) => abandoned += 1,
        }
    }
    let last = top.into_sorted();
    let frac = abandoned as f64 / cands.len() as f64;

    let t = Instant::now();
    let reps = 5;
    let mut sink = Vec::new();
    for _ in 0..reps {
        sink = rerank_early_abandon_dyn(q, vecs, cands, k, stride);
    }
    let _ = sink;
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    (secs, last, frac)
}

fn main() {
    let dim = 768;
    let n_vectors = 100_000;
    let n_queries = 32;
    let k = 10;

    println!("=== ruvector-partial-l2 benchmark ===");
    println!(
        "dim={dim} n_vectors={n_vectors} n_queries={n_queries} k={k}  (release, single-thread)"
    );

    println!("generating vectors...");
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|i| make(1000 + i as u64, dim))
        .collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|i| make(i as u64 * 31 + 7, dim)).collect();
    let cands: Vec<u32> = (0..n_vectors as u32).collect();

    // Sanity: kernel micro-benchmark (single-pair, 1M reps).
    println!("\n--- single-pair kernel throughput (dim={dim}, 1e6 reps) ---");
    {
        let a = &vectors[0];
        let b = &vectors[1];
        let reps = 1_000_000u64;

        let t = Instant::now();
        let mut acc = 0.0f32;
        for _ in 0..reps {
            acc += l2_full(a, b);
        }
        let dt = t.elapsed().as_secs_f64();
        let gflops = (reps as f64 * dim as f64 * 3.0) / dt / 1e9;
        println!(
            "  l2_full            : {:>8.3} ns/call  ({:.2} GFLOPS)  sink={acc}",
            dt / reps as f64 * 1e9,
            gflops
        );

        let full = l2_full(a, b);
        for &stride in &[32usize, 64, 128] {
            let t = Instant::now();
            let mut sink = 0.0f32;
            // Threshold well above true distance so we time the
            // non-abandoning path (worst case for early abandon).
            let tau = full * 2.0;
            for _ in 0..reps {
                let r = match stride {
                    32 => l2_early_abandon::<32>(a, b, tau),
                    64 => l2_early_abandon::<64>(a, b, tau),
                    128 => l2_early_abandon::<128>(a, b, tau),
                    _ => unreachable!(),
                };
                if let AbandonResult::Full(v) = r {
                    sink += v;
                }
            }
            let dt = t.elapsed().as_secs_f64();
            println!(
                "  l2_ea<{stride:>3}> (worst): {:>8.3} ns/call  sink={sink}",
                dt / reps as f64 * 1e9,
            );
        }
    }

    // Rerank scenario.
    println!("\n--- rerank scenario ({n_queries} queries × {n_vectors} candidates) ---");
    let mut times_full = Vec::with_capacity(n_queries);
    let mut baseline_ids: Vec<Vec<u32>> = Vec::with_capacity(n_queries);
    for q in &queries {
        let (secs, top) = measure_full(q, &vectors, &cands, k);
        times_full.push(secs);
        baseline_ids.push(top.iter().map(|x| x.1).collect());
    }
    let median_full = median(&mut times_full.clone());
    let total_full: f64 = times_full.iter().sum();
    println!(
        "  baseline (full L2)         : total={:>7.3}s  median/query={:>7.3}ms",
        total_full,
        median_full * 1e3
    );

    for &stride in &[32usize, 64, 128] {
        let mut times = Vec::with_capacity(n_queries);
        let mut fracs = Vec::with_capacity(n_queries);
        let mut recall_ok = true;
        for (qi, q) in queries.iter().enumerate() {
            let (secs, top, frac) = measure_ea(q, &vectors, &cands, k, stride);
            let ea_ids: Vec<u32> = top.iter().map(|x| x.1).collect();
            if ea_ids != baseline_ids[qi] {
                recall_ok = false;
            }
            times.push(secs);
            fracs.push(frac);
        }
        let med = median(&mut times.clone());
        let total: f64 = times.iter().sum();
        let mean_frac: f64 = fracs.iter().sum::<f64>() / fracs.len() as f64;
        let speedup = total_full / total;
        println!(
            "  early-abandon dyn stride={stride:>3}  : total={:>7.3}s  median/query={:>7.3}ms  \
             abandoned={:>5.1}%  speedup={:>4.2}x  recall={}",
            total,
            med * 1e3,
            mean_frac * 100.0,
            speedup,
            if recall_ok { "1.000" } else { "MISMATCH" }
        );
        assert!(recall_ok, "recall parity violated at dyn stride={stride}");
    }

    // Const-generic path — the compiler can fully unroll the inner
    // block, which is the recommended production path.
    println!("  --- const-generic (recommended production path) ---");
    for stride in [32usize, 64, 128] {
        let t = Instant::now();
        let reps = 5;
        let mut last = Vec::new();
        for _ in 0..reps {
            for q in &queries {
                last = match stride {
                    32 => rerank_early_abandon::<32>(q, &vectors, &cands, k),
                    64 => rerank_early_abandon::<64>(q, &vectors, &cands, k),
                    128 => rerank_early_abandon::<128>(q, &vectors, &cands, k),
                    _ => unreachable!(),
                };
            }
        }
        let _ = last;
        let total = t.elapsed().as_secs_f64() / reps as f64;
        let med = total / n_queries as f64;
        let speedup = total_full / total;
        // Verify recall.
        let mut recall_ok = true;
        for (qi, q) in queries.iter().enumerate() {
            let top = match stride {
                32 => rerank_early_abandon::<32>(q, &vectors, &cands, k),
                64 => rerank_early_abandon::<64>(q, &vectors, &cands, k),
                128 => rerank_early_abandon::<128>(q, &vectors, &cands, k),
                _ => unreachable!(),
            };
            let ids: Vec<u32> = top.iter().map(|x| x.1).collect();
            if ids != baseline_ids[qi] {
                recall_ok = false;
            }
        }
        println!(
            "  early-abandon const stride={stride:>3}: total={:>7.3}s  median/query={:>7.3}ms  \
             speedup={:>4.2}x  recall={}",
            total,
            med * 1e3,
            speedup,
            if recall_ok { "1.000" } else { "MISMATCH" }
        );
        assert!(recall_ok, "recall parity violated at const stride={stride}");
    }

    println!("\n(recall parity is exact by construction — abandonment is triggered only\n\
              when the running partial squared L2 already exceeds the current top-k\n\
              worst distance.)");
}
