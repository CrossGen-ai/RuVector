use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use ruvector_cpolytope_lsh_ivf::{
    CosineRanker, CrossPolytopeRanker, L2Ranker, ProbeRanker,
};

fn rand_centroids(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / n).collect()
        })
        .collect()
}

#[test]
fn hash_determinism_under_seed() {
    let centroids = rand_centroids(64, 32, 11);
    let a = CrossPolytopeRanker::new(&centroids, 12, 7);
    let b = CrossPolytopeRanker::new(&centroids, 12, 7);
    let q = rand_centroids(1, 32, 99).pop().unwrap();
    let ra = a.rank(&q, &centroids);
    let rb = b.rank(&q, &centroids);
    // Scores should be bit-identical given the same seed.
    assert_eq!(ra.len(), rb.len());
    for (x, y) in ra.iter().zip(rb.iter()) {
        assert_eq!(x.0, y.0);
        assert_eq!(x.1.to_bits(), y.1.to_bits(), "ranker not deterministic");
    }
}

#[test]
fn cosine_ranker_is_invariant_to_query_norm() {
    let centroids = rand_centroids(32, 16, 1);
    let r = CosineRanker::new(&centroids);
    let q = rand_centroids(1, 16, 2).pop().unwrap();
    let q_big: Vec<f32> = q.iter().map(|x| x * 1_000.0).collect();
    let o1 = r.rank(&q, &centroids);
    let o2 = r.rank(&q_big, &centroids);
    let order1: Vec<usize> = o1.into_iter().map(|p| p.0).collect();
    let order2: Vec<usize> = o2.into_iter().map(|p| p.0).collect();
    assert_eq!(order1, order2, "cosine order changed with query magnitude");
}

#[test]
fn cross_polytope_order_invariant_to_query_norm() {
    let centroids = rand_centroids(32, 16, 3);
    let r = CrossPolytopeRanker::new(&centroids, 16, 101);
    let q = rand_centroids(1, 16, 4).pop().unwrap();
    let q_big: Vec<f32> = q.iter().map(|x| x * 1_000.0).collect();
    let o1: Vec<usize> = r.rank(&q, &centroids).into_iter().map(|p| p.0).collect();
    let o2: Vec<usize> = r.rank(&q_big, &centroids).into_iter().map(|p| p.0).collect();
    assert_eq!(o1, o2, "cross-polytope order changed with query magnitude");
}

/// Smoke recall test: compared to a random probe order, cross-polytope LSH
/// should clearly do better at recovering the nearest centroid at modest
/// nprobe. We check that recall@16 > 0.5 over a batch, well above the
/// random baseline of `16 / 128 = 0.125`.
#[test]
fn cross_polytope_beats_random_probe_order_at_nprobe16() {
    let dim = 32;
    let n_centroids = 128;
    let centroids = rand_centroids(n_centroids, dim, 777);
    let r = CrossPolytopeRanker::new(&centroids, 16, 555);

    let mut rng = ChaCha12Rng::seed_from_u64(31);
    let mut hits = 0usize;
    let trials = 200usize;
    for _ in 0..trials {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let q: Vec<f32> = v.into_iter().map(|x| x / n).collect();

        // Truth = nearest centroid by cosine.
        let mut best = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (i, c) in centroids.iter().enumerate() {
            let s: f32 = q.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            if s > best_s { best_s = s; best = i; }
        }

        let order = r.rank(&q, &centroids);
        let top16: std::collections::HashSet<usize> =
            order.into_iter().take(16).map(|(i, _)| i).collect();
        if top16.contains(&best) { hits += 1; }
    }
    let recall = hits as f32 / trials as f32;
    assert!(recall > 0.5, "recall@16 was only {:.3}; expected > 0.5", recall);
}

#[test]
fn l2_and_cosine_agree_on_unit_sphere_order() {
    // On unit-norm centroids, -||q-c||_2^2 and <q,c> are monotone-related,
    // so L2 ordering and Cosine ordering should match exactly.
    let dim = 24;
    let centroids = rand_centroids(50, dim, 9);
    let q = rand_centroids(1, dim, 10).pop().unwrap();
    let o_l2: Vec<usize> = L2Ranker.rank(&q, &centroids).into_iter().map(|p| p.0).collect();
    let o_cos: Vec<usize> = CosineRanker::new(&centroids).rank(&q, &centroids).into_iter().map(|p| p.0).collect();
    assert_eq!(o_l2, o_cos);
}
