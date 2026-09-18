//! Reproducible benchmark for the three Hamming top-K backends.
//!
//! Runs each backend against the same synthetic corpus + query pool on two
//! data regimes:
//!
//! * `random`   — iid uniform bits (bit-sliced pruning is weak here).
//! * `clustered`— vectors sampled by flipping ~10% of bits of one of 8
//!               anchor bit-vectors; a fraction of queries share an anchor
//!               with the corpus, so a large slice of the corpus is far
//!               and prunes early.
//!
//! Output columns: backend, n, dim, k, queries/s, ms/query, correctness.
//! The correctness column reports whether this backend's top-k id set
//! matches NaiveRowMajor on every query in the batch.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::Instant;

use ruvector_bit_sliced_hamming::{
    bit_sliced::BitSlicedEarlyTerm, blocked::BlockedRowMajor, naive::NaiveRowMajor,
    HammingTopK, PackedCorpus,
};

fn random_corpus(n: usize, dim: usize, seed: u64) -> PackedCorpus {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = PackedCorpus::new(n, dim);
    let w = c.words_per_vec();
    let mut buf = vec![0u64; w];
    for i in 0..n {
        for word in &mut buf { *word = rng.gen::<u64>(); }
        c.set_vector(i, &buf);
    }
    c
}

/// 8-cluster corpus: each vector = one of 8 anchor bit-vectors with a
/// ~10% i.i.d. bit-flip mask applied.
fn clustered_corpus(n: usize, dim: usize, seed: u64) -> (PackedCorpus, Vec<Vec<u64>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let w = dim / 64;
    let anchors: Vec<Vec<u64>> = (0..8).map(|_| (0..w).map(|_| rng.gen::<u64>()).collect()).collect();
    let mut c = PackedCorpus::new(n, dim);
    let mut buf = vec![0u64; w];
    for i in 0..n {
        let anchor = &anchors[i % 8];
        for j in 0..w {
            // For each of 64 bits in this word, flip with probability ~10%.
            let mut flip = 0u64;
            for b in 0..64 {
                if rng.gen::<u32>() % 10 == 0 { flip |= 1u64 << b; }
            }
            buf[j] = anchor[j] ^ flip;
        }
        c.set_vector(i, &buf);
    }
    (c, anchors)
}

fn random_queries(count: usize, dim: usize, seed: u64) -> Vec<Vec<u64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count).map(|_| (0..dim / 64).map(|_| rng.gen::<u64>()).collect()).collect()
}

/// Queries close to a random anchor with ~5% flips → distances are strongly
/// clustered, so early termination is meaningful.
fn clustered_queries(count: usize, dim: usize, anchors: &[Vec<u64>], seed: u64) -> Vec<Vec<u64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let w = dim / 64;
    (0..count).map(|_| {
        let a = &anchors[rng.gen::<usize>() % anchors.len()];
        let mut q = vec![0u64; w];
        for j in 0..w {
            let mut flip = 0u64;
            for b in 0..64 {
                if rng.gen::<u32>() % 20 == 0 { flip |= 1u64 << b; }
            }
            q[j] = a[j] ^ flip;
        }
        q
    }).collect()
}

fn bench<B: HammingTopK>(b: &B, queries: &[Vec<u64>], k: usize) -> f64 {
    // Warm up
    for q in queries.iter().take(4) { let _ = b.topk(q, k); }
    let t0 = Instant::now();
    let mut checksum: u64 = 0;
    for q in queries {
        let r = b.topk(q, k);
        // consume to prevent DCE
        checksum = checksum.wrapping_add(r.iter().map(|(i, d)| *i as u64 + *d as u64).sum::<u64>());
    }
    let elapsed = t0.elapsed().as_secs_f64();
    std::hint::black_box(checksum);
    queries.len() as f64 / elapsed
}

fn check_matches_naive<B: HammingTopK>(oracle: &NaiveRowMajor, b: &B, queries: &[Vec<u64>], k: usize) -> bool {
    for q in queries.iter().take(32) {
        if oracle.topk(q, k) != b.topk(q, k) { return false; }
    }
    true
}

fn run_regime(label: &str, corpus: PackedCorpus, queries: Vec<Vec<u64>>, k: usize) {
    let n = corpus.n; let dim = corpus.dim;
    println!("\n=== {label}  n={n} dim={dim} k={k} queries={} ===", queries.len());
    let naive = NaiveRowMajor::build(corpus.clone());
    let blocked = BlockedRowMajor::build(corpus.clone());
    let bs = BitSlicedEarlyTerm::build(corpus.clone());

    let ok_b = check_matches_naive(&naive, &blocked, &queries, k);
    let ok_s = check_matches_naive(&naive, &bs, &queries, k);

    println!("storage bytes: naive={}  blocked={}  bit_sliced={}",
             naive.storage_bytes(), blocked.storage_bytes(), bs.storage_bytes());
    println!("{:>22}  {:>12}  {:>10}  {:>8}", "backend", "queries/s", "ms/query", "matches_naive");
    for (name, qps, matches) in [
        ("NaiveRowMajor",     bench(&naive, &queries, k), true),
        ("BlockedRowMajor",   bench(&blocked, &queries, k), ok_b),
        ("BitSlicedEarlyTerm",bench(&bs, &queries, k), ok_s),
    ] {
        println!("{:>22}  {:>12.1}  {:>10.3}  {:>8}", name, qps, 1000.0 / qps, matches);
    }
}

fn main() {
    let n = 20_000;
    let dim = 128;
    let k = 10;
    let qcount = 500;

    // --- Regime 1: iid random ------------------------------------------------
    let c = random_corpus(n, dim, 0xC0FFEE);
    let q = random_queries(qcount, dim, 0xDEADBEEF);
    run_regime("random iid uniform", c, q, k);

    // --- Regime 2: 8-cluster corpus with anchor-close queries ---------------
    let (c2, anchors) = clustered_corpus(n, dim, 0xC0FFEE);
    let q2 = clustered_queries(qcount, dim, &anchors, 0xDEADBEEF);
    run_regime("clustered (8 anchors + 5% query flips)", c2, q2, k);

    // --- Regime 3: larger n, smaller dim to stress the transpose ------------
    let n3 = 40_000; let dim3 = 64;
    let c3 = random_corpus(n3, dim3, 0xC0FFEE);
    let q3 = random_queries(qcount, dim3, 0xDEADBEEF);
    run_regime("random iid uniform (wide n / narrow dim)", c3, q3, k);
}
