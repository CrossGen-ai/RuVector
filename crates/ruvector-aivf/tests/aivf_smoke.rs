//! Lightweight self-bench runnable as `cargo test --release -p ruvector-aivf
//! --test aivf_bench -- --nocapture`.  Criterion was avoided to keep the
//! crate dependency-light; the demo bin (`aivf-demo`) is the primary harness.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_aivf::{Aivf, AivfConfig, FlatQuantizer};

#[test]
fn build_and_search_smoke() {
    let dim = 32; let n = 2_000;
    let mut rng = StdRng::seed_from_u64(7);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let mut cfg = AivfConfig::new(dim, 16);
    cfg.nprobe = 4;
    cfg.rebalance_every = 256;
    let idx = Aivf::build(cfg, &data, FlatQuantizer::new(dim));
    let q: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let res = idx.search(&q, 5);
    assert_eq!(res.len(), 5);
    // Distances must be monotonically non-decreasing.
    for w in res.windows(2) {
        assert!(w[0].1 <= w[1].1, "{:?}", res);
    }
}

#[test]
fn split_increases_list_count_on_drift() {
    let dim = 16;
    let mut rng = StdRng::seed_from_u64(123);
    let mut cfg = AivfConfig::new(dim, 4);
    cfg.split_size = 64;          // force splits early
    cfg.rebalance_every = 64;
    cfg.merge_size = 2;
    // Bootstrap from one tight cluster.
    let boot: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 0.1).collect())
        .collect();
    let mut idx = Aivf::build(cfg, &boot, FlatQuantizer::new(dim));
    let start_lists = idx.num_lists();
    // Stream a totally different region — should trigger splits.
    for i in 0..1000 {
        let v: Vec<f32> = (0..dim).map(|_| 10.0 + rng.gen::<f32>() * 0.1).collect();
        idx.add((200 + i) as u32, &v);
    }
    assert!(idx.num_lists() > start_lists,
            "expected splits; before={} after={}", start_lists, idx.num_lists());
    assert!(idx.split_events() > 0);
}
