//! End-to-end demo. Build a small IVF, hammer a hotspot, watch cells split.

use rand::{rngs::StdRng, Rng, SeedableRng};
use ruvector_crackivf_adaptive::{
    brute_topk, recall_at_k, AdaptiveThreshold, FixedThreshold, IvfIndex, NoSplit,
};

fn gen(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn run<P: ruvector_crackivf_adaptive::SplitPolicy>(
    name: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    policy: P,
    nlist_0: usize,
    nprobe: usize,
) {
    let mut idx = IvfIndex::build(vectors.to_vec(), nlist_0, policy, 42).unwrap();
    let start_cells = idx.nlist();
    let mut recall_sum = 0.0f32;
    for q in queries {
        let truth = brute_topk(vectors, q, 10);
        let got = idx.search(q, 10, nprobe).unwrap();
        recall_sum += recall_at_k(&got, &truth);
    }
    let recall = recall_sum / queries.len() as f32;
    let s = idx.stats();
    println!(
        "{name:<22}  nlist {}→{}  splits={:>3}  probes={:>5}  recall@10={:.3}  bytes={}",
        start_cells,
        idx.nlist(),
        s.splits,
        s.total_probes,
        recall,
        idx.estimated_bytes()
    );
}

fn main() {
    let vectors = gen(2000, 32, 1);
    // Skewed query workload: 70% of queries hit a small hotspot region.
    let mut rng = StdRng::seed_from_u64(2);
    let queries: Vec<Vec<f32>> = (0..200)
        .map(|i| {
            if i % 10 < 7 {
                // hotspot
                let base = &vectors[10 + (i % 20)];
                base.iter().map(|x| x + rng.gen::<f32>() * 0.02 - 0.01).collect()
            } else {
                (0..32).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
            }
        })
        .collect();

    println!("=== CrackIVF demo — 2000 vectors × 32 dims, 200 skewed queries ===");
    run("baseline (NoSplit)", &vectors, &queries, NoSplit, 8, 2);
    run(
        "fixed threshold",
        &vectors,
        &queries,
        FixedThreshold::new(4, 32),
        8,
        2,
    );
    run(
        "adaptive threshold",
        &vectors,
        &queries,
        AdaptiveThreshold::new(6, 2, 32),
        8,
        2,
    );
}
