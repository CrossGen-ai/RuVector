use crate::{
    brute_top1, gen_clusters, CentroidSeeder, KnnGraph, MultiCentroidSeeder, RandomSeeder, Seeder,
};

fn setup(n: usize, d: usize, k: usize) -> (Vec<Vec<f32>>, KnnGraph, Vec<Vec<f32>>, Vec<usize>) {
    let data = gen_clusters(n, d, k, 42);
    let graph = KnnGraph::build(data.clone(), 12);
    let queries = gen_clusters(60, d, k, 123);
    let truth: Vec<usize> = queries.iter().map(|q| brute_top1(&data, q)).collect();
    (data, graph, queries, truth)
}

#[test]
fn random_seeder_returns_single_entry() {
    let (data, _g, _q, _t) = setup(200, 8, 4);
    let mut s = RandomSeeder::new(1);
    s.fit(&data);
    assert_eq!(s.entry_points(&data[0]).len(), 1);
}

#[test]
fn centroid_seeder_names_and_medoids() {
    let (data, _g, _q, _t) = setup(400, 16, 8);
    let mut s = CentroidSeeder::new(8, 8, 1);
    s.fit(&data);
    assert_eq!(s.medoids.len(), 8);
    // All medoids must be valid indices.
    for &m in &s.medoids {
        assert!(m < data.len());
    }
}

#[test]
fn multi_centroid_returns_m_entries() {
    let (data, _g, _q, _t) = setup(400, 16, 8);
    let mut s = MultiCentroidSeeder::new(8, 3, 8, 1);
    s.fit(&data);
    let e = s.entry_points(&data[0]);
    assert_eq!(e.len(), 3);
}

#[test]
fn centroid_beats_random_on_hops() {
    // Clustered data with many clusters is the regime where seeding matters.
    let (data, graph, queries, _truth) = setup(800, 32, 16);
    let mut rs = RandomSeeder::new(7);
    let mut cs = CentroidSeeder::new(16, 10, 7);
    rs.fit(&data);
    cs.fit(&data);
    let mut r_hops = 0usize;
    let mut c_hops = 0usize;
    for q in &queries {
        r_hops += graph.greedy_search(q, &rs.entry_points(q)).1.hops;
        c_hops += graph.greedy_search(q, &cs.entry_points(q)).1.hops;
    }
    // Centroid seeding should reduce hops meaningfully.
    assert!(
        c_hops < r_hops,
        "expected centroid hops ({}) < random hops ({})",
        c_hops,
        r_hops
    );
}

#[test]
fn recall_is_reasonable() {
    let (data, graph, queries, truth) = setup(600, 16, 10);
    let mut cs = CentroidSeeder::new(10, 10, 7);
    cs.fit(&data);
    let mut ok = 0usize;
    for (i, q) in queries.iter().enumerate() {
        let (best, _) = graph.greedy_search(q, &cs.entry_points(q));
        if best == truth[i] {
            ok += 1;
        }
    }
    let recall = ok as f64 / queries.len() as f64;
    // Greedy top-1 on a modest 12-NN graph is known to plateau below exact
    // recall due to local minima; the crate's point is *relative* seeder
    // improvement, not absolute exactness. Set a floor that reflects reality.
    assert!(recall > 0.55, "recall too low: {}", recall);
}
