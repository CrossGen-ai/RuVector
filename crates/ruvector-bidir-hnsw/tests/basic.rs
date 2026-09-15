use ruvector_bidir_hnsw::bench::{default_config, gen_dataset, run_bench};
use ruvector_bidir_hnsw::build::{build_hnsw_like, BuildParams};
use ruvector_bidir_hnsw::graph::l2_sq;
use ruvector_bidir_hnsw::search::{search, Mode};

#[test]
fn graph_is_connected_enough_to_search() {
    let mut cfg = default_config();
    cfg.n = 500;
    cfg.n_queries = 20;
    let (vectors, queries) = gen_dataset(&cfg);
    let g = build_hnsw_like(
        &vectors,
        BuildParams { m_max: cfg.m_max, ef_construction: cfg.ef_construction },
    );
    for q in &queries {
        let (top, _s) = search(&g, &vectors, q, cfg.k, cfg.ef, Mode::Baseline);
        assert_eq!(top.len(), cfg.k);
        // returned distances must be non-decreasing.
        for w in top.windows(2) {
            assert!(w[0].0 <= w[1].0);
        }
    }
}

#[test]
fn reverse_expansion_never_reduces_recall() {
    let mut cfg = default_config();
    cfg.n = 1500;
    cfg.n_queries = 60;
    let results = run_bench(&cfg);
    let base = results.iter().find(|r| r.mode == "baseline").unwrap();
    let rev = results.iter().find(|r| r.mode == "expand-reverse").unwrap();
    // Reverse expansion should retrieve at least as many true neighbors as
    // baseline, since it only adds candidates.
    assert!(
        rev.recall_at_k + 1e-9 >= base.recall_at_k,
        "reverse recall {} < baseline {}",
        rev.recall_at_k,
        base.recall_at_k
    );
}

#[test]
fn exact_topk_matches_bruteforce() {
    use ruvector_bidir_hnsw::bench::exact_topk;
    let mut cfg = default_config();
    cfg.n = 200;
    let (vectors, queries) = gen_dataset(&cfg);
    let q = &queries[0];
    let top = exact_topk(&vectors, q, 5);
    // Recompute distances and confirm order.
    let mut d: Vec<(f32, usize)> =
        vectors.iter().enumerate().map(|(i, v)| (l2_sq(q, v), i)).collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for i in 0..5 {
        assert_eq!(top[i], d[i].1);
    }
}
