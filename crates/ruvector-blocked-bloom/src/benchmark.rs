//! Real benchmark: HNSW-shaped visited-set workload across three impls.
//!
//! Workload model:
//! - Graph of `n_ids` nodes (default 1,000,000).
//! - Per query: pick a random seed and walk a frontier of `frontier` candidates,
//!   each with `degree` neighbors.  Repeat for `hops` layers.  This is a good
//!   proxy for HNSW / Vamana beam search where most `mark` calls hit the same
//!   candidate cluster.
//! - Every neighbor id is `mark`'d.  ~35% of ids repeat within a query
//!   (matches the DBpedia-1M measurement from the roargraph paper).
//!
//! We measure wall-clock throughput per impl, memory footprint, and — for the
//! Bloom variant — the observed false-positive rate against an oracle set.
//!
//! Reproducible: `cargo run --release -p ruvector-blocked-bloom --bin benchmark`.

use rand::{Rng, SeedableRng, rngs::StdRng};
use ruvector_blocked_bloom::{
    BitmapVisited, BlockedBloomVisited, BloomStats, HashSetVisited, VisitedSet,
};
use std::collections::HashSet;
use std::time::Instant;

struct Config {
    n_ids: u32,
    frontier: usize,
    degree: usize,
    hops: usize,
    n_queries: usize,
    seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_ids: 1_000_000,
            frontier: 64,
            degree: 32,
            hops: 8,
            n_queries: 2_000,
            seed: 0xB100_D_C0FFEE,
        }
    }
}

fn gen_query(cfg: &Config, rng: &mut StdRng) -> Vec<u32> {
    // Simulate HNSW beam: cluster of ids drawn from a moving window with
    // ~35% overlap.  This gives realistic collision pressure.
    let mut ids = Vec::with_capacity(cfg.frontier * cfg.degree * cfg.hops);
    let mut center: u32 = rng.gen_range(0..cfg.n_ids);
    for _ in 0..cfg.hops {
        for _ in 0..cfg.frontier {
            for _ in 0..cfg.degree {
                // Local jitter around center, with occasional teleport.
                let jitter: i64 = rng.gen_range(-2048..2048);
                let id = (center as i64 + jitter).rem_euclid(cfg.n_ids as i64) as u32;
                ids.push(id);
            }
        }
        // Move the center to simulate progression toward the query.
        let step: i64 = rng.gen_range(-8192..8192);
        center = (center as i64 + step).rem_euclid(cfg.n_ids as i64) as u32;
    }
    ids
}

fn bench<V: VisitedSet>(
    label: &str,
    mut set: V,
    queries: &[Vec<u32>],
) -> (f64, usize, u64) {
    // Warm up (touch every block/page once) and drop the sample.
    for q in queries.iter().take(4) {
        for &id in q {
            set.mark(id);
        }
        set.reset();
    }

    let bytes = set.bytes();
    let mut new_count = 0u64;
    let start = Instant::now();
    for q in queries {
        for &id in q {
            if set.mark(id) {
                new_count += 1;
            }
        }
        set.reset();
    }
    let elapsed = start.elapsed();
    let total_ops: u64 = queries.iter().map(|q| q.len() as u64).sum();
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    println!(
        "  {label:<28} {bytes:>10} B   {mops:>7.2} Mops/s   {ns:>6.1} ns/op   new={new_count}",
        mops = ops_per_sec / 1_000_000.0,
        ns = elapsed.as_nanos() as f64 / total_ops as f64,
    );
    (ops_per_sec, bytes, new_count)
}

/// Measure the observed false-positive rate of the Bloom variant against a
/// ground-truth oracle set.  Because HNSW cares about "would this cause me to
/// skip a real neighbor?", the metric is: fraction of `mark` calls where
/// the Bloom said "seen" but the oracle said "new".
fn bloom_fp_rate(cfg: &Config, queries: &[Vec<u32>]) -> (f64, BloomStats) {
    let mut bloom = BlockedBloomVisited::for_load(cfg.frontier * cfg.degree * cfg.hops);
    let mut oracle: HashSet<u32> = HashSet::with_capacity(4096);

    let mut fp = 0u64;
    let mut probes = 0u64;
    let mut last_stats = bloom.stats();
    for q in queries {
        for &id in q {
            let bloom_new = bloom.mark(id);
            let oracle_new = oracle.insert(id);
            probes += 1;
            // False positive = Bloom said "already seen" but oracle said "new".
            if !bloom_new && oracle_new {
                fp += 1;
            }
        }
        last_stats = bloom.stats();
        bloom.reset();
        oracle.clear();
    }
    (fp as f64 / probes as f64, last_stats)
}

fn main() {
    let cfg = Config::default();
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    println!("ruvector-blocked-bloom :: HNSW-shape visited-set benchmark");
    println!(
        "  n_ids={} frontier={} degree={} hops={} queries={}",
        cfg.n_ids, cfg.frontier, cfg.degree, cfg.hops, cfg.n_queries
    );
    println!();

    let queries: Vec<Vec<u32>> = (0..cfg.n_queries).map(|_| gen_query(&cfg, &mut rng)).collect();
    let total_ops: u64 = queries.iter().map(|q| q.len() as u64).sum();
    println!("  total probes = {total_ops}");
    println!();

    println!("Variant                          Memory        Throughput         Latency");
    println!("---------------------------------------------------------------------------");
    let (hs_ops, _, _) = bench(
        "HashSet<u32>",
        HashSetVisited::with_capacity(cfg.frontier * cfg.degree),
        &queries,
    );
    let (bm_ops, _, _) = bench(
        "Bitmap<u64>",
        BitmapVisited::new(cfg.n_ids as usize),
        &queries,
    );
    let (bb_ops, _, _) = bench(
        "BlockedBloom(512b, k=4)",
        BlockedBloomVisited::for_load(cfg.frontier * cfg.degree * cfg.hops),
        &queries,
    );

    println!();
    let (fp, stats) = bloom_fp_rate(&cfg, &queries);
    println!(
        "BlockedBloom FP rate: {:.4}%  (dirty_blocks/query ~= {}, n_blocks={})",
        fp * 100.0,
        stats.dirty_blocks,
        stats.n_blocks
    );

    println!();
    println!("Speedups vs HashSet<u32> baseline:");
    println!("  Bitmap<u64>            : {:.2}x", bm_ops / hs_ops);
    println!("  BlockedBloom(512b, k=4): {:.2}x", bb_ops / hs_ops);
}
