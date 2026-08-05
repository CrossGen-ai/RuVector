//! Bench binary: measures MIPS recall@10, encode/query throughput, and memory
//! for isotropic vs anisotropic PQ under identical corpus + hyperparameters.
//!
//! Real numbers only — no simulation, no fake timings. Run with:
//! `cargo run --release -p ruvector-anisotropic-pq --bin aniso-pq-bench`.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use ruvector_anisotropic_pq::{
    recall_at_k, AnisoPqIndex, AnisotropicTrainer, IsotropicTrainer, MipsResult,
    PqTrainConfig,
};

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn gauss(rng: &mut StdRng) -> f32 {
    // Box-Muller
    let u1: f32 = rng.gen_range(1e-6..1.0);
    let u2: f32 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

fn make_corpus(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    // Anisotropic-Gaussian corpus: low-rank signal + isotropic noise. This is
    // a first-order model of learned dense embeddings (BERT/E5/ANCE), which
    // concentrate energy in a low-dim subspace and carry a tail of isotropic
    // noise. It is the setting where score-aware quantization is expected to
    // outperform reconstruction-optimal quantization.
    let mut rng = StdRng::seed_from_u64(seed);
    let rank = (d / 4).max(4);
    // Fixed random basis (independent of `seed` so all corpora share geometry).
    let mut basis_rng = StdRng::seed_from_u64(0xB1A5_5EED);
    let basis: Vec<Vec<f32>> = (0..rank)
        .map(|_| {
            let v: Vec<f32> = (0..d).map(|_| gauss(&mut basis_rng)).collect();
            let n2: f32 = v.iter().map(|x| x * x).sum();
            let inv = 1.0 / n2.sqrt().max(1e-6);
            v.into_iter().map(|x| x * inv).collect()
        })
        .collect();
    (0..n)
        .map(|_| {
            let z: Vec<f32> = (0..rank).map(|_| gauss(&mut rng)).collect();
            let noise: Vec<f32> = (0..d).map(|_| 0.15 * gauss(&mut rng)).collect();
            let mut x = noise;
            for (k, b) in basis.iter().enumerate() {
                let zk = z[k];
                for j in 0..d {
                    x[j] += zk * b[j];
                }
            }
            x
        })
        .collect()
}

fn ground_truth_mips(data: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> =
        data.iter().enumerate().map(|(i, v)| (i, dot(v, q))).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

struct VariantReport {
    name: String,
    build_ms: f32,
    query_us_mean: f32,
    recall_at_10: f32,
    memory_kb: f32,
}

fn bench_variant(
    label: &str,
    idx: &AnisoPqIndex,
    queries: &[Vec<f32>],
    truth: &[Vec<usize>],
    build_ms: f32,
) -> VariantReport {
    let k = 10usize;
    let mut recall_sum = 0.0f32;
    let t = Instant::now();
    for (i, q) in queries.iter().enumerate() {
        let res: Vec<MipsResult> = idx.search(q, k);
        let ids: Vec<usize> = res.into_iter().map(|r| r.id).collect();
        recall_sum += recall_at_k(&ids, &truth[i], k);
    }
    let elapsed = t.elapsed().as_secs_f32() * 1e6;
    VariantReport {
        name: label.to_string(),
        build_ms,
        query_us_mean: elapsed / queries.len() as f32,
        recall_at_10: recall_sum / queries.len() as f32,
        memory_kb: idx.memory_bytes() as f32 / 1024.0,
    }
}

fn main() {
    // Corpus: 8k × 64 (small enough to run in ~seconds without a release
    // matrix lib but big enough for meaningful recall differences).
    let n = std::env::var("ANISO_N").ok().and_then(|s| s.parse().ok()).unwrap_or(8_192usize);
    let d = std::env::var("ANISO_D").ok().and_then(|s| s.parse().ok()).unwrap_or(64usize);
    let q_count = 200usize;

    println!("== Anisotropic PQ bench ==");
    println!("N={n} D={d} Q={q_count}   (env: ANISO_N, ANISO_D)");

    let corpus = make_corpus(n, d, 0xC01D_BEEF);
    let queries = make_corpus(q_count, d, 0xF00D_D00D);

    // Ground truth MIPS.
    let t = Instant::now();
    let truth: Vec<Vec<usize>> =
        queries.iter().map(|q| ground_truth_mips(&corpus, q, 10)).collect();
    let gt_ms = t.elapsed().as_secs_f32() * 1e3;
    println!("Ground-truth MIPS built in {gt_ms:.1} ms");

    let cfg = PqTrainConfig { m: 8, k: 256, iters: 20, seed: 0xA150 };
    println!(
        "PQ config: M={} K={} iters={} sub-dim={}",
        cfg.m,
        cfg.k,
        cfg.iters,
        d / cfg.m
    );

    let mut reports = vec![];

    // Variant 1: isotropic baseline.
    let t = Instant::now();
    let iso = AnisoPqIndex::build(&IsotropicTrainer, &corpus, &corpus, &cfg).unwrap();
    let build_ms = t.elapsed().as_secs_f32() * 1e3;
    reports.push(bench_variant("isotropic (η=1)", &iso, &queries, &truth, build_ms));

    // Sweep η ∈ {2, 4, 8, 16} to trace the recall vs anisotropy curve.
    for &eta in &[2.0f32, 4.0, 8.0, 16.0] {
        let t = Instant::now();
        let idx = AnisoPqIndex::build(
            &AnisotropicTrainer::new(eta),
            &corpus,
            &corpus,
            &cfg,
        )
        .unwrap();
        let build_ms = t.elapsed().as_secs_f32() * 1e3;
        let label = format!("anisotropic η={:>4.0}", eta);
        reports.push(bench_variant(&label, &idx, &queries, &truth, build_ms));
    }

    println!();
    println!("{:<22} {:>10} {:>12} {:>12} {:>10}",
        "variant", "build_ms", "query_us", "recall@10", "mem_kb");
    println!("{}", "-".repeat(72));
    for r in &reports {
        println!(
            "{:<22} {:>10.1} {:>12.2} {:>12.3} {:>10.1}",
            r.name, r.build_ms, r.query_us_mean, r.recall_at_10, r.memory_kb
        );
    }
    // Simple derived metrics for the doc.
    let iso_r = reports[0].recall_at_10;
    for r in reports.iter().skip(1) {
        let delta = r.recall_at_10 - iso_r;
        println!(
            "  Δrecall({} vs isotropic) = {:+.3}",
            r.name, delta
        );
    }
}
