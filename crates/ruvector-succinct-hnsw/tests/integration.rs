//! End-to-end integration test.
//!
//! Invariants:
//!   1. All three backends decode the same neighbour set for every node.
//!   2. Recall on the encoded backends equals recall on the dense
//!      baseline (encoding is lossless).
//!   3. `reordered_delta_varbyte` reports fewer payload bytes than the
//!      dense baseline on a clustered corpus (the target workload).

use ruvector_succinct_hnsw::adjacency::{
    Adjacency, DeltaVarByteAdj, DenseAdj, ReorderedDeltaAdj,
};
use ruvector_succinct_hnsw::bench::{run, BenchConfig};
use ruvector_succinct_hnsw::graph::{build_graph, BuildParams};
use ruvector_succinct_hnsw::reorder::bfs_permutation;
use ruvector_succinct_hnsw::{NodeId, SqEuclid, Vector};

fn small_corpus() -> Vec<Vector> {
    // 200 pts on a 2D grid — dense enough to give the graph structure.
    let mut v = Vec::new();
    for i in 0..20i32 {
        for j in 0..10i32 {
            v.push(vec![i as f32, j as f32]);
        }
    }
    v
}

#[test]
fn all_backends_agree_per_node() {
    let corpus = small_corpus();
    let lists = build_graph(
        &corpus,
        &SqEuclid,
        BuildParams { m: 8, ef_construction: 32 },
    );
    let dense = DenseAdj::from_lists(lists.clone());
    let delta = DeltaVarByteAdj::from_lists(lists.clone());
    let perm = bfs_permutation(&lists, 2);
    let reord = ReorderedDeltaAdj::from_lists_with_permutation(lists.clone(), perm);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for v in 0..dense.len() as NodeId {
        dense.neighbors_into(v, &mut a);
        delta.neighbors_into(v, &mut b);
        reord.neighbors_into(v, &mut c);
        a.sort_unstable();
        b.sort_unstable();
        c.sort_unstable();
        assert_eq!(a, b, "delta backend differs at node {v}");
        assert_eq!(a, c, "reordered backend differs at node {v}");
    }
}

#[test]
fn encodings_preserve_recall_and_shrink_memory() {
    let cfg = BenchConfig {
        n: 2_000,
        dim: 16,
        n_queries: 40,
        m: 16,
        ef_construction: 64,
        ef_search: 96,
        k: 10,
        n_clusters: 32,
        clustered: true,
        seed: 42,
    };
    let r = run(cfg);
    assert_eq!(r.variants.len(), 3);
    let base = &r.variants[0];
    let delta = &r.variants[1];
    let reord = &r.variants[2];
    // Recall must be identical — same graph, same search.
    assert!((base.avg_recall - delta.avg_recall).abs() < 1e-6);
    assert!((base.avg_recall - reord.avg_recall).abs() < 1e-6);
    // Reordered must shrink payload on the clustered target.
    assert!(
        reord.bytes < base.bytes,
        "reordered {} !< baseline {}",
        reord.bytes,
        base.bytes
    );
    // Sanity: recall is meaningfully non-zero on a well-formed graph.
    assert!(base.avg_recall > 0.5, "unexpectedly low recall {}", base.avg_recall);
}
