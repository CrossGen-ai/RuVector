use ruvector_centroid_seeded_hnsw::{
    brute_top1, gen_clusters, CentroidSeeder, KnnGraph, MultiCentroidSeeder, RandomSeeder, Rng,
    Seeder,
};

fn main() {
    let n = 2_000;
    let d = 32;
    let clusters = 20;
    let data = gen_clusters(n, d, clusters, 42);
    println!("Building k-NN graph over N={} D={} (this is O(N^2 D))...", n, d);
    let graph = KnnGraph::build(data.clone(), 16);

    // Seeders
    let mut random_s = RandomSeeder::new(7);
    let mut cent_s = CentroidSeeder::new(clusters, 12, 7);
    let mut mcent_s = MultiCentroidSeeder::new(clusters, 4, 12, 7);
    random_s.fit(&data);
    cent_s.fit(&data);
    mcent_s.fit(&data);

    // Queries: draw fresh points from the same generative model.
    let queries = gen_clusters(200, d, clusters, 9999);
    let truth: Vec<usize> = queries.iter().map(|q| brute_top1(&data, q)).collect();

    for s in [
        &random_s as &dyn Seeder,
        &cent_s as &dyn Seeder,
        &mcent_s as &dyn Seeder,
    ] {
        let mut correct = 0usize;
        let mut hops = 0usize;
        let mut dc = 0usize;
        for (qi, q) in queries.iter().enumerate() {
            let entries = s.entry_points(q);
            let (best, st) = graph.greedy_search(q, &entries);
            if best == truth[qi] {
                correct += 1;
            }
            hops += st.hops;
            dc += st.distance_calls;
        }
        println!(
            "seeder={:>10}  recall@1={:.3}  avg_hops={:.2}  avg_dist_calls={:.2}",
            s.name(),
            correct as f64 / queries.len() as f64,
            hops as f64 / queries.len() as f64,
            dc as f64 / queries.len() as f64,
        );
    }
    // touch Rng so it isn't dead code in downstream tools
    let _ = Rng::new(1).next_u64();
}
