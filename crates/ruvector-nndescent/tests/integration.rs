//! Integration tests — real algorithms, real numbers, no mocks.
//!
//! Acceptance contract for NN-Descent LocalJoin:
//!   - graph recall@K >= 0.95 vs brute force on clustered data
//!   - distance ops < 50% of brute force at N >= 1000

use ruvector_nndescent::{
    BruteForceBuilder, Dataset, DistanceCounter, KnnGraphBuilder, NnDescentBuilder, NnDescentConfig,
    SeededHnsw, SeededHnswConfig,
};

fn small_clustered() -> Dataset {
    Dataset::synthetic_clustered(1_200, 32, 12, 50, 7)
}

#[test]
fn brute_force_self_recall_is_one() {
    let ds = small_clustered();
    let c = DistanceCounter::new();
    let g = BruteForceBuilder.build(&ds.vectors, 10, &c);
    assert_eq!(g.recall_against(&g, 10), 1.0);
    assert!(c.count() > 0);
}

#[test]
fn nndescent_localjoin_meets_recall_floor() {
    let ds = small_clustered();
    let bf_c = DistanceCounter::new();
    let truth = BruteForceBuilder.build(&ds.vectors, 10, &bf_c);

    let cfg = NnDescentConfig { rho: 0.5, delta: 1e-3, max_iters: 20, use_reverse: true, seed: 11 };
    let nd_c = DistanceCounter::new();
    let g = NnDescentBuilder::new(cfg).build(&ds.vectors, 10, &nd_c);

    let r = g.recall_against(&truth, 10);
    assert!(r >= 0.95, "NN-Descent LocalJoin graph recall@10 = {r:.3} (need >= 0.95)");

    let ratio = (nd_c.count() as f64) / (bf_c.count() as f64);
    assert!(ratio < 0.5, "NN-Descent did {ratio:.3}x brute distance ops (need < 0.5)");
}

#[test]
fn nndescent_basic_is_faster_than_brute_in_ops() {
    let ds = small_clustered();
    let bf_c = DistanceCounter::new();
    let _ = BruteForceBuilder.build(&ds.vectors, 10, &bf_c);

    let cfg = NnDescentConfig { rho: 0.5, delta: 1e-3, max_iters: 12, use_reverse: false, seed: 12 };
    let nd_c = DistanceCounter::new();
    let _ = NnDescentBuilder::new(cfg).build(&ds.vectors, 10, &nd_c);

    assert!(
        nd_c.count() < bf_c.count(),
        "basic NN-Descent ops {} should be < brute ops {}",
        nd_c.count(), bf_c.count()
    );
}

#[test]
fn seeded_hnsw_search_is_consistent_across_builders() {
    // Same graph quality => same search recall (within noise). This is the
    // navigability-equivalence guarantee that justifies bulk construction.
    let ds = small_clustered();
    let k = 10;

    let truth = BruteForceBuilder.build(&ds.vectors, k, &DistanceCounter::new());
    let nd_g = NnDescentBuilder::new(NnDescentConfig {
        rho: 0.5, delta: 1e-3, max_iters: 20, use_reverse: true, seed: 99,
    })
    .build(&ds.vectors, k, &DistanceCounter::new());

    let entries = SeededHnswConfig::spread_entries(ds.vectors.len(), 8);

    let bf_search = SeededHnsw::new(&ds.vectors, &truth, SeededHnswConfig {
        ef_search: 48, entry_points: entries.clone(),
    });
    let nd_search = SeededHnsw::new(&ds.vectors, &nd_g, SeededHnswConfig {
        ef_search: 48, entry_points: entries,
    });

    // Run a single deterministic query through both; result sets must
    // overlap heavily (this is a navigability sanity check, not a
    // recall-ceiling test).
    let q = &ds.queries[0];
    let a = bf_search.search(q, k);
    let b = nd_search.search(q, k);
    let set_a: std::collections::HashSet<u32> = a.into_iter().collect();
    let overlap = b.into_iter().filter(|x| set_a.contains(x)).count();
    assert!(
        overlap >= k / 2,
        "seeded HNSW on NN-Descent graph diverges from brute graph: overlap = {overlap}/{k}"
    );
}
