//! End-to-end demo across two graph regimes:
//!   * pure exact-kNN substrate (no long edges)
//!   * small-world-ish substrate (4 of m slots are random long edges)
//! Each regime measures recall, hops, distance computations, and the entry
//! distance^2 for the three strategies.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

use ruvector_lsh_seeded_hnsw_entry::{
    EntryStrategy, LshSeedIndex, Nsw, NswParams, SearchParams,
};
use ruvector_lsh_seeded_hnsw_entry::search::{brute_force_topk, search};

fn gen_centers(dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..clusters * dim)
        .map(|_| {
            let x: f32 = rng.sample(StandardNormal);
            x * 5.0
        })
        .collect()
}

fn clustered_from(centers: &[f32], n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = i % clusters;
        for d in 0..dim {
            let noise: f32 = rng.sample(StandardNormal);
            out.push(centers[c * dim + d] + noise);
        }
    }
    out
}

fn evaluate(index: &Nsw, lsh: &LshSeedIndex, queries: &[f32], dim: usize, k: usize, ef: usize) {
    let qn = queries.len() / dim;
    let ground: Vec<Vec<u32>> = (0..qn)
        .map(|q| brute_force_topk(&index.vectors, dim, &queries[q * dim..(q + 1) * dim], k))
        .collect();
    let strategies: &[(&str, EntryStrategy)] = &[
        ("fixed ", EntryStrategy::Fixed),
        ("random", EntryStrategy::Random),
        ("lsh   ", EntryStrategy::Lsh(lsh)),
    ];
    let params = SearchParams { ef, k };
    let mut rng = StdRng::seed_from_u64(99);
    println!("strategy  recall@{}   hops/q   distcomp/q  entry_d^2   us/q", k);
    for (name, strat) in strategies {
        let mut hits = 0u64;
        let mut hops = 0u64;
        let mut dc = 0u64;
        let mut ed = 0.0f64;
        let t = std::time::Instant::now();
        for q in 0..qn {
            let (topk, stats) = search(
                index,
                &queries[q * dim..(q + 1) * dim],
                params,
                strat,
                &mut rng,
            );
            let truth = &ground[q];
            for id in &topk {
                if truth.contains(id) {
                    hits += 1;
                }
            }
            hops += stats.hops;
            dc += stats.distance_computations;
            ed += stats.entry_distance as f64;
        }
        let dt = t.elapsed();
        let qf = qn as f64;
        println!(
            "{}    {:.4}   {:>6.1}   {:>8.1}    {:>8.2}   {:>5.2}",
            name,
            hits as f64 / (qn as f64 * k as f64),
            hops as f64 / qf,
            dc as f64 / qf,
            ed / qf,
            dt.as_secs_f64() * 1e6 / qf,
        );
    }
}

fn main() {
    const N: usize = 20_000;
    const DIM: usize = 64;
    const CLUSTERS: usize = 32;
    const QUERIES: usize = 500;
    const K: usize = 10;
    const EF: usize = 64;
    const M: usize = 24;
    const LSH_BITS: u8 = 12;

    println!("# ruvector-lsh-seeded-hnsw-entry demo");
    println!(
        "n={} dim={} clusters={} queries={} k={} ef={} m={}",
        N, DIM, CLUSTERS, QUERIES, K, EF, M
    );

    let centers = gen_centers(DIM, CLUSTERS, 7);
    let vectors = clustered_from(&centers, N, DIM, CLUSTERS, 11);
    let queries = clustered_from(&centers, QUERIES, DIM, CLUSTERS, 42);

    // -------- Regime A: pure exact-kNN substrate --------
    println!("\n## Regime A: pure exact-kNN graph (no long edges)");
    let t = std::time::Instant::now();
    let index_a = Nsw::build(
        vectors.clone(),
        DIM,
        NswParams { m: M, ef_construction: 128, long_edges: 0, long_edge_seed: 0 },
    );
    println!("graph built in {:.2}s", t.elapsed().as_secs_f32());
    let lsh_a = LshSeedIndex::build(&index_a, LSH_BITS, 1234);
    println!("lsh built: {} buckets ({} bits)\n", lsh_a.bucket_count(), lsh_a.bits);
    evaluate(&index_a, &lsh_a, &queries, DIM, K, EF);

    // -------- Regime B: small-world (4 random long edges per node) --------
    println!("\n## Regime B: small-world (4 long edges per node)");
    let t = std::time::Instant::now();
    let index_b = Nsw::build(
        vectors,
        DIM,
        NswParams { m: M, ef_construction: 128, long_edges: 4, long_edge_seed: 0xC0FFEE },
    );
    println!("graph built in {:.2}s", t.elapsed().as_secs_f32());
    let lsh_b = LshSeedIndex::build(&index_b, LSH_BITS, 1234);
    println!("lsh built: {} buckets ({} bits)\n", lsh_b.bucket_count(), lsh_b.bits);
    evaluate(&index_b, &lsh_b, &queries, DIM, K, EF);
}
