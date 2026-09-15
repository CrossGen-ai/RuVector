use ruvector_bidir_hnsw::bench::{default_config, gen_dataset, exact_topk};
use ruvector_bidir_hnsw::build::{build_hnsw_like, BuildParams};
use ruvector_bidir_hnsw::search::{search, Mode};

fn main() {
    let mut cfg = default_config();
    cfg.n = 1000; cfg.n_queries = 10;
    let (v, q) = gen_dataset(&cfg);
    let g = build_hnsw_like(&v, BuildParams{m_max: cfg.m_max, ef_construction: cfg.ef_construction});
    // degree stats
    let mut deg_hist = vec![0; cfg.m_max+1];
    for u in 0..g.n { let d = g.neighbors(u).count(); deg_hist[d] += 1; }
    println!("degree hist: {:?}", deg_hist);
    // per query with huge ef
    for (qi, qv) in q.iter().enumerate().take(3) {
        let gt = exact_topk(&v, qv, 10);
        for &ef in &[32usize, 128, 512, 2000] {
            let (top, s) = search(&g, &v, qv, 10, ef, Mode::Baseline);
            let hits = top.iter().filter(|(_,i)| gt.contains(i)).count();
            let top_dists: Vec<f32> = top.iter().map(|(d,_)| *d).collect();
            let gt_d0 = ruvector_bidir_hnsw::graph::l2_sq(qv, &v[gt[0]]);
            let gt_d9 = ruvector_bidir_hnsw::graph::l2_sq(qv, &v[gt[9]]);
            println!("q{} ef={} hits={}/10 evals={} visit={} top_d[0]={:.3} top_d[9]={:.3} gt_d[0]={:.3} gt_d[9]={:.3}",
                qi, ef, hits, s.distance_evals, s.nodes_visited, top_dists[0], top_dists[9], gt_d0, gt_d9);
        }
    }
}
