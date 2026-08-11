use ruvector_hnsw_rng_diverse::{build_index, data, search_multi, AlphaPrune, Naive, Pruner, RngPrune};

fn brute_topk(vs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, f32)> = vs.iter().enumerate()
        .map(|(i, v)| (i, ruvector_hnsw_rng_diverse::dist2(v, q))).collect();
    d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    d.into_iter().take(k).map(|(i, _)| i).collect()
}

fn eval(vs: &[Vec<f32>], name: &str, pruner: &dyn Pruner, m: usize, ef_build: usize, queries: &[Vec<f32>], gt: &[Vec<usize>], ef_search: usize) {
    let t = std::time::Instant::now();
    let idx = build_index(vs, pruner, m, ef_build);
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = std::time::Instant::now();
    let mut hits1 = 0usize;
    let mut recall10 = 0.0f64;
    let mut hops = 0usize;
    let mut calls = 0usize;
    // Diverse multi-entry: evenly-spaced ids as a cheap proxy for cluster restart.
    let entries: Vec<usize> = (0..8).map(|k| k * (vs.len() / 8)).collect();
    for (qi, q) in queries.iter().enumerate() {
        let (r, s) = search_multi(&idx, vs, &entries, q, 10, ef_search);
        hops += s.hops;
        calls += s.dist_calls;
        if r[0].0 == gt[qi][0] { hits1 += 1; }
        let overlap = r.iter().filter(|(id, _)| gt[qi].contains(id)).count();
        recall10 += overlap as f64 / 10.0;
    }
    let query_us = t.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
    println!(
        "{:>14} | deg={:5.2} | build={:7.1} ms | r@1={:.3} | r@10={:.3} | ef_s={:>3} | hops={:5.1} | calls={:6.1} | {:6.1} us/q",
        name, idx.avg_degree, build_ms,
        hits1 as f64 / queries.len() as f64,
        recall10 / queries.len() as f64,
        ef_search,
        hops as f64 / queries.len() as f64,
        calls as f64 / queries.len() as f64,
        query_us,
    );
}

fn main() {
    let n = 1500;
    let d = 32;
    let c = 12;
    let m = 12;
    let ef_build = 40;
    let vs = data::mixture(n, d, c, 42);
    // Queries: perturbed base vectors.
    let mut rng = data::Rng::new(9001);
    let queries: Vec<Vec<f32>> = (0..200).map(|_| {
        let src = (rng.next_u64() as usize) % n;
        (0..d).map(|j| vs[src][j] + rng.normal() * 0.1).collect()
    }).collect();
    let gt: Vec<Vec<usize>> = queries.iter().map(|q| brute_topk(&vs, q, 10)).collect();

    println!("== HNSW-style graph, neighbor-selection comparison ==");
    println!("N={n} D={d} clusters={c} M={m} ef_build={ef_build}, 200 queries\n");
    for ef_s in [8usize, 32, 128] {
        eval(&vs, "naive-topM",   &Naive,             m, ef_build, &queries, &gt, ef_s);
        eval(&vs, "rng-prune",    &RngPrune,          m, ef_build, &queries, &gt, ef_s);
        eval(&vs, "alpha=1.2",    &AlphaPrune::new(1.2), m, ef_build, &queries, &gt, ef_s);
        eval(&vs, "alpha=1.5",    &AlphaPrune::new(1.5), m, ef_build, &queries, &gt, ef_s);
        println!();
    }
}
