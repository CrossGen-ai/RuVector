//! Minimal demo: build a 1k x 32-dim graph, compare all three searchers
//! against brute-force on 50 queries. Run: `cargo run -p ruvector-tribase --example tribase_demo`.

use ruvector_tribase::*;

fn main() {
    let n = 1000;
    let d = 32;
    let m = 16;
    let g = FlatGraph::random_knn(n, d, m, 42);

    let base = BaselineSearcher { graph: &g, entry: 0 };
    let tri1 = TribaseSearcher::build(&g, 0);
    let trik = MultiLandmarkSearcher::build(&g, 0, 8, 99);

    let k = 10;
    let ef = 64;
    let q_count = 50;

    for searcher in [
        &base as &dyn AnnSearcher,
        &tri1,
        &trik,
    ] {
        let mut total = SearchStats::default();
        let mut recall = 0.0f64;
        for qi in 0..q_count as u32 {
            let q = g.vector(qi).to_vec();
            let truth = brute_force(&g, &q, k);
            let truth_ids: std::collections::HashSet<u32> = truth.iter().map(|(i, _)| *i).collect();
            let mut s = SearchStats::default();
            let r = searcher.search(&q, k, ef, &mut s);
            total.merge(&s);
            let hits = r.iter().filter(|(i, _)| truth_ids.contains(i)).count();
            recall += hits as f64 / k as f64;
        }
        recall /= q_count as f64;
        println!(
            "{:>10} | recall@{}={:.3} | full_dist/query={:.1} | pruned/query={:.1} | visited/query={:.1}",
            searcher.name(),
            k,
            recall,
            total.full_dist as f64 / q_count as f64,
            total.pruned as f64 / q_count as f64,
            total.visited as f64 / q_count as f64,
        );
    }
}
