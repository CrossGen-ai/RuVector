use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use ruvector_blockmax_ivf::{BlockMaxTightScanner, IvfPqBuilder, PqParams, Scanner, SequentialScanner, l2_sq};

fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

#[test]
fn tight_block_max_matches_sequential() {
    // The provably-sound (tight) block-max scan must return the exact same
    // candidate set as the sequential scan for every query, by construction.
    let dim = 32;
    let n = 2_000;
    let data = gen_data(n, dim, 1);
    let index = IvfPqBuilder {
        nlist: 16, kmeans_iter: 6, block_size: 32,
        pq: PqParams { m: 8, ksub: 64, kmeans_iter: 6 },
        seed: 7,
    }.train(&data, dim);

    let queries = gen_data(50, dim, 99);
    let mut total_skipped = 0u64;
    let mut total_blocks = 0u64;
    for qi in 0..50 {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let (seq, _) = SequentialScanner.search(&index, q, 10, 16);
        let (tight, stats) = BlockMaxTightScanner.search(&index, q, 10, 16);
        assert_eq!(seq.len(), tight.len());
        for i in 0..seq.len() {
            assert_eq!(seq[i].id, tight[i].id, "query {qi} rank {i}: expected {} got {}", seq[i].id, tight[i].id);
            assert!((seq[i].dist - tight[i].dist).abs() < 1e-5);
        }
        total_skipped += stats.blocks_skipped;
        total_blocks += stats.blocks_total;
    }
    // Over 50 queries the tight bound should prune at least some blocks.
    assert!(total_skipped > 0, "tight block-max skipped {}/{} blocks across 50 queries", total_skipped, total_blocks);
}

#[test]
fn ivf_pq_recall_is_sane() {
    let dim = 32;
    let n = 2_000;
    let data = gen_data(n, dim, 2);
    let index = IvfPqBuilder {
        nlist: 16, kmeans_iter: 10, block_size: 32,
        pq: PqParams { m: 8, ksub: 128, kmeans_iter: 10 },
        seed: 7,
    }.train(&data, dim);

    let queries = gen_data(30, dim, 42);
    let mut hits = 0;
    let total = 30 * 10;
    for qi in 0..30 {
        let q = &queries[qi * dim..(qi + 1) * dim];
        // ground truth = linear scan on raw vectors
        let mut gt: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, l2_sq(q, &data[i as usize * dim..(i as usize + 1) * dim]))).collect();
        gt.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt_top: std::collections::HashSet<u32> = gt.into_iter().take(10).map(|(i, _)| i).collect();
        let (seq, _) = SequentialScanner.search(&index, q, 10, 16);
        for h in seq { if gt_top.contains(&h.id) { hits += 1; } }
    }
    let recall = hits as f32 / total as f32;
    assert!(recall > 0.5, "recall@10 was {recall}, expected > 0.5 for nprobe=all");
}
