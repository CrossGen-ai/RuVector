//! Timing-only bench: holds dataset + graph fixed, varies entry strategy.
//! No criterion dependency — std::time is sufficient to publish real numbers.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rand::Rng;

use ruvector_lsh_seeded_hnsw_entry::{EntryStrategy, LshSeedIndex, Nsw, NswParams, SearchParams};
use ruvector_lsh_seeded_hnsw_entry::search::search;

fn gen_centers(dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = Vec::with_capacity(clusters * dim);
    for _ in 0..clusters * dim {
        let x: f32 = rng.sample(StandardNormal);
        centers.push(x * 5.0);
    }
    centers
}

fn make(centers: &[f32], n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
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

fn main() {
    let configs = [(20_000usize, 64usize, 48usize), (50_000, 96, 64)];
    for (n, dim, clusters) in configs {
        println!("\n=== n={} dim={} clusters={} ===", n, dim, clusters);
        let centers = gen_centers(dim, clusters, 7);
        let vectors = make(&centers, n, dim, clusters, 11);
        let queries = make(&centers, 2000, dim, clusters, 777);
        let index = Nsw::build(vectors, dim, NswParams { m: 16, ef_construction: 64, long_edges: 4, long_edge_seed: 0xC0FFEE });
        let lsh = LshSeedIndex::build(&index, 14, 2024);

        let params = SearchParams { ef: 48, k: 10 };
        let mut rng = StdRng::seed_from_u64(0);

        for (name, strat) in [
            ("fixed ", EntryStrategy::Fixed),
            ("random", EntryStrategy::Random),
            ("lsh   ", EntryStrategy::Lsh(&lsh)),
        ] {
            // Warm up.
            for q in 0..64 {
                let _ = search(&index, &queries[q * dim..(q + 1) * dim], params, &strat, &mut rng);
            }
            let t = std::time::Instant::now();
            let mut hops = 0u64;
            let mut dc = 0u64;
            for q in 0..2000 {
                let (_, s) = search(&index, &queries[q * dim..(q + 1) * dim], params, &strat, &mut rng);
                hops += s.hops;
                dc += s.distance_computations;
            }
            let dt = t.elapsed();
            println!(
                "{} :: {:>5.2} us/q, hops/q {:>5.1}, distcomp/q {:>6.1}",
                name,
                dt.as_secs_f64() * 1e6 / 2000.0,
                hops as f64 / 2000.0,
                dc as f64 / 2000.0,
            );
        }
    }
}
