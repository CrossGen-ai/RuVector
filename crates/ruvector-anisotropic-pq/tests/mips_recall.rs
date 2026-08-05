//! End-to-end sanity: on a modest MIPS corpus, anisotropic PQ should match or
//! exceed isotropic PQ at recall@10 for the same code budget.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use ruvector_anisotropic_pq::{
    recall_at_k, AnisoPqIndex, AnisotropicTrainer, IsotropicTrainer, PqTrainConfig,
};

fn gauss(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen_range(1e-6..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn corpus(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = StdRng::seed_from_u64(seed);
    (0..n).map(|_| (0..d).map(|_| gauss(&mut r)).collect()).collect()
}

fn top_mips(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut s: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()))
        .collect();
    s.sort_by(|a, b| b.1.total_cmp(&a.1));
    s.into_iter().take(k).map(|(i, _)| i).collect()
}

#[test]
fn anisotropic_matches_or_beats_isotropic_recall_at_10() {
    let data = corpus(2048, 32, 11);
    let queries = corpus(50, 32, 22);
    let truth: Vec<Vec<usize>> = queries.iter().map(|q| top_mips(&data, q, 10)).collect();
    let cfg = PqTrainConfig { m: 4, k: 64, iters: 10, seed: 33 };

    let iso = AnisoPqIndex::build(&IsotropicTrainer, &data, &data, &cfg).unwrap();
    let ani = AnisoPqIndex::build(&AnisotropicTrainer::new(8.0), &data, &data, &cfg).unwrap();

    let recall = |idx: &AnisoPqIndex| -> f32 {
        let mut s = 0.0f32;
        for (i, q) in queries.iter().enumerate() {
            let approx: Vec<usize> =
                idx.search(q, 10).into_iter().map(|r| r.id).collect();
            s += recall_at_k(&approx, &truth[i], 10);
        }
        s / queries.len() as f32
    };
    let r_iso = recall(&iso);
    let r_ani = recall(&ani);
    assert!(r_iso > 0.05, "iso recall too low: {r_iso}");
    // Allow small slack for RNG jitter; the whole point is anisotropic should
    // not lose on MIPS.
    assert!(
        r_ani + 0.02 >= r_iso,
        "anisotropic ({r_ani}) unexpectedly worse than isotropic ({r_iso})"
    );
}
