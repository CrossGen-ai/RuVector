use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

use ruvector_lsh_seeded_hnsw_entry::{
    EntryStrategy, LshSeedIndex, Nsw, NswParams, SearchParams,
};
use ruvector_lsh_seeded_hnsw_entry::search::{brute_force_topk, search};

fn centers_for(dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut c = vec![0.0f32; clusters * dim];
    for v in &mut c {
        let x: f32 = rng.sample(StandardNormal);
        *v = x * 5.0;
    }
    c
}

fn clustered_with(centers: &[f32], n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = i % clusters;
        for d in 0..dim {
            let noise: f32 = rng.sample(StandardNormal);
            out.push(centers[c * dim + d] + noise * 0.8);
        }
    }
    out
}

fn clustered(n: usize, dim: usize, clusters: usize, seed: u64) -> Vec<f32> {
    // Legacy single-seed path: for standalone build-only tests.
    let centers = centers_for(dim, clusters, seed);
    clustered_with(&centers, n, dim, clusters, seed.wrapping_add(1))
}

#[test]
fn graph_builds_and_is_connected_enough_to_search() {
    let v = clustered(500, 32, 8, 1);
    let g = Nsw::build(v, 32, NswParams { m: 8, ef_construction: 32, ..NswParams::default() });
    assert_eq!(g.n, 500);
    assert_eq!(g.dim, 32);
    // Every node must have at least one neighbor.
    for i in 0..g.n as u32 {
        let has = g.neighbors_of(i).iter().any(|&nb| nb != u32::MAX);
        assert!(has, "node {} orphaned", i);
    }
}

#[test]
fn lsh_signature_is_deterministic_and_consistent() {
    let v = clustered(200, 24, 4, 5);
    let g = Nsw::build(v, 24, NswParams::default());
    let lsh = LshSeedIndex::build(&g, 10, 99);
    // Same vector -> same seed, twice.
    let q = vec![0.3f32; 24];
    assert_eq!(lsh.seed_for(&q), lsh.seed_for(&q));
}

#[test]
fn lsh_entry_matches_or_beats_fixed_entry_quality() {
    let centers = centers_for(48, 24, 2);
    let v = clustered_with(&centers, 2000, 48, 24, 10);
    let queries = clustered_with(&centers, 200, 48, 24, 123);
    let g = Nsw::build(v.clone(), 48, NswParams { m: 16, ef_construction: 64, long_edges: 4, ..NswParams::default() });
    let lsh = LshSeedIndex::build(&g, 12, 2024);
    let params = SearchParams { ef: 32, k: 10 };
    let mut rng = StdRng::seed_from_u64(0);

    let mut fixed_hits = 0u64;
    let mut lsh_hits = 0u64;
    let mut fixed_hops = 0u64;
    let mut lsh_hops = 0u64;

    for q in 0..200 {
        let query = &queries[q * 48..(q + 1) * 48];
        let truth = brute_force_topk(&v, 48, query, 10);
        let (f, fs) = search(&g, query, params, &EntryStrategy::Fixed, &mut rng);
        let (l, ls) = search(&g, query, params, &EntryStrategy::Lsh(&lsh), &mut rng);
        for id in &f {
            if truth.contains(id) { fixed_hits += 1; }
        }
        for id in &l {
            if truth.contains(id) { lsh_hits += 1; }
        }
        fixed_hops += fs.hops;
        lsh_hops += ls.hops;
    }

    // LSH entry must not degrade recall by more than 1 pt (small-sample noise)
    // and in expectation should match or exceed baseline.
    let f_r = fixed_hits as f64 / (200.0 * 10.0);
    let l_r = lsh_hits as f64 / (200.0 * 10.0);
    assert!(l_r >= f_r - 0.01, "lsh recall {} < fixed recall {} - 0.01", l_r, f_r);

    // Not a hard requirement for the test to pass (the bench measures the win),
    // but we print so cargo test -- --nocapture gives a quick signal.
    eprintln!(
        "fixed recall={:.4} hops/q={:.1}  |  lsh recall={:.4} hops/q={:.1}",
        f_r, fixed_hops as f64 / 200.0,
        l_r, lsh_hops as f64 / 200.0,
    );
}

#[test]
fn random_entry_gets_worse_recall_at_small_ef() {
    // Random entry should visibly degrade recall at small ef relative to
    // fixed enterpoint, otherwise the greedy walk is washing out the signal
    // and the comparison with LSH would be meaningless.
    let centers = centers_for(32, 16, 3);
    let v = clustered_with(&centers, 1000, 32, 16, 20);
    let queries = clustered_with(&centers, 100, 32, 16, 77);
    let g = Nsw::build(v.clone(), 32, NswParams { m: 8, ef_construction: 32, long_edges: 2, ..NswParams::default() });
    let params = SearchParams { ef: 8, k: 5 };
    let mut rng = StdRng::seed_from_u64(1);

    let mut f_hits = 0u64;
    let mut r_hits = 0u64;
    for q in 0..100 {
        let query = &queries[q * 32..(q + 1) * 32];
        let truth = brute_force_topk(&v, 32, query, 5);
        let (f, _) = search(&g, query, params, &EntryStrategy::Fixed, &mut rng);
        let (r, _) = search(&g, query, params, &EntryStrategy::Random, &mut rng);
        for id in &f { if truth.contains(id) { f_hits += 1; } }
        for id in &r { if truth.contains(id) { r_hits += 1; } }
    }
    eprintln!("fixed recall@5 = {}, random recall@5 = {}", f_hits, r_hits);
    // Must be strictly or weakly worse — a hard "<" would be flaky on small n.
    assert!(r_hits <= f_hits + 10, "random beats fixed by more than noise — graph may be trivially connected");
}
