//! Numeric acceptance: on a clustered embedding-like corpus (representative of
//! real text/image embeddings), MultiProbeSimHash with L=8 tables and 8 probes
//! must reach recall@10 >= 0.60.

use ruvector_lsh::*;

#[test]
fn multiprobe_recall_floor_clustered() {
    let n = 5_000;
    let dim = 64;
    let nq = 100;
    let k = 10;

    // 50 clusters, sigma=0.15 — moderate cluster separation.
    let data = gen_clustered_dataset(n, dim, 50, 0.15, 11);
    // Queries are near-neighbours of random corpus points.
    let queries = gen_queries_near(&data, nq, 0.10, 13);

    let bf = BruteForce::new(data.clone());
    let truth: Vec<_> = queries.iter().map(|q| bf.search(q, k)).collect();

    let idx = MultiProbeSimHash::build(data.clone(), 32, 8, 8, 99);

    let mut total = 0.0f32;
    for (qi, q) in queries.iter().enumerate() {
        let res = idx.search(q, k);
        total += recall_at_k(&res, &truth[qi], k);
    }
    let mean = total / nq as f32;
    println!("MultiProbeSimHash recall@10 = {mean:.3}");
    assert!(
        mean >= 0.60,
        "MultiProbeSimHash recall@10 too low: {mean:.3} (need >= 0.60)"
    );
}
