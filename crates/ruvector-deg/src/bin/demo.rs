//! Minimal `cargo run --release -p ruvector-deg --bin deg-demo`
//! Prints recall and edge counts for the three backends on a tiny corpus.

use rand::prelude::*;
use ruvector_deg::{AnnIndex, DegConfig, DegGraph, NswGraph, RandomGraph, metric::sq_l2};

fn main() {
    let n = 2_000usize;
    let dim = 32usize;
    let k = 10usize;
    let ef = 64usize;

    let mut rng = StdRng::seed_from_u64(42);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect();

    let mut rg = RandomGraph::new(dim, 16, 7);
    let mut ns = NswGraph::new(dim, 16, 64);
    let mut cfg = DegConfig::default();
    cfg.degree = 16; cfg.ef_construction = 64; cfg.optimize_every = 256; cfg.optimize_passes = 1;
    let mut dg = DegGraph::new(dim, cfg);

    for v in &data {
        rg.insert(v.clone());
        ns.insert(v.clone());
        dg.insert(v.clone());
    }
    let final_swaps = dg.optimize_all(2);

    let (mut r_rand, mut r_nsw, mut r_deg) = (0.0, 0.0, 0.0);
    for q in &queries {
        let mut gold: Vec<(u32, f32)> = data.iter().enumerate()
            .map(|(i, v)| (i as u32, sq_l2(q, v))).collect();
        gold.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let g: std::collections::HashSet<u32> = gold.iter().take(k).map(|(i, _)| *i).collect();
        let hit = |got: Vec<(u32, f32)>| -> f32 {
            got.iter().filter(|(i, _)| g.contains(i)).count() as f32 / k as f32
        };
        r_rand += hit(rg.search(q, k, ef));
        r_nsw  += hit(ns.search(q, k, ef));
        r_deg  += hit(dg.search(q, k, ef));
    }
    let qf = queries.len() as f32;
    println!("N={n} dim={dim} k={k} ef={ef}");
    println!("recall@{k}  random = {:.3}  ({} edges)", r_rand / qf, rg.edge_count());
    println!("recall@{k}  nsw    = {:.3}  ({} edges)", r_nsw  / qf, ns.edge_count());
    println!("recall@{k}  deg    = {:.3}  ({} edges, {} final swaps)", r_deg / qf, dg.edge_count(), final_swaps);
}
