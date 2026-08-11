use ruvector_hnsw_rng_diverse::{build_index, data, dist2, search_multi, AlphaPrune, Naive, Pruner, RngPrune};

fn brute_topk(vs: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut d: Vec<(usize, f32)> = vs.iter().enumerate().map(|(i, v)| (i, dist2(v, q))).collect();
    d.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    d.into_iter().take(k).map(|(i, _)| i).collect()
}

fn bench_one(label: &str, vs: &[Vec<f32>], pruner: &dyn Pruner, m: usize, ef_build: usize, queries: &[Vec<f32>], gt: &[Vec<usize>]) {
    let t = std::time::Instant::now();
    let idx = build_index(vs, pruner, m, ef_build);
    let build_ms = t.elapsed().as_secs_f64() * 1000.0;
    for &ef_s in &[8usize, 32, 128] {
        let t = std::time::Instant::now();
        let (mut r1, mut r10, mut hops, mut calls) = (0usize, 0.0f64, 0usize, 0usize);
        let entries: Vec<usize> = (0..8).map(|k| k * (vs.len() / 8)).collect();
        for (qi, q) in queries.iter().enumerate() {
            let (r, s) = search_multi(&idx, vs, &entries, q, 10, ef_s);
            hops += s.hops; calls += s.dist_calls;
            if r[0].0 == gt[qi][0] { r1 += 1; }
            let overlap = r.iter().filter(|(id, _)| gt[qi].contains(id)).count();
            r10 += overlap as f64 / 10.0;
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        println!("{:>12} | {:>10} | build={:7.1}ms | ef_s={:>3} | deg={:5.2} | r@1={:.3} | r@10={:.3} | hops={:5.1} | calls={:6.1} | {:6.1}us/q",
            label, pruner.name(), build_ms, ef_s, idx.avg_degree,
            r1 as f64 / queries.len() as f64,
            r10 / queries.len() as f64,
            hops as f64 / queries.len() as f64,
            calls as f64 / queries.len() as f64,
            us);
    }
}

fn main() {
    println!("== ruvector-hnsw-rng-diverse — neighbor-selection benchmark ==\n");
    let configs = [(1000usize, 32usize, 10usize, 12usize, 40usize),
                   (2500,      64,      16,      16,      64)];
    for &(n, d, c, m, ef_build) in &configs {
        let vs = data::mixture(n, d, c, 42);
        let mut rng = data::Rng::new(9001);
        let queries: Vec<Vec<f32>> = (0..200).map(|_| {
            let src = (rng.next_u64() as usize) % n;
            (0..d).map(|j| vs[src][j] + rng.normal() * 0.1).collect()
        }).collect();
        let gt: Vec<Vec<usize>> = queries.iter().map(|q| brute_topk(&vs, q, 10)).collect();
        let label = format!("N={n} D={d}");
        println!("-- {label} clusters={c} M={m} ef_b={ef_build} --");
        bench_one(&label, &vs, &Naive,              m, ef_build, &queries, &gt);
        bench_one(&label, &vs, &RngPrune,           m, ef_build, &queries, &gt);
        bench_one(&label, &vs, &AlphaPrune::new(1.2), m, ef_build, &queries, &gt);
        bench_one(&label, &vs, &AlphaPrune::new(1.5), m, ef_build, &queries, &gt);
        println!();
    }
}
