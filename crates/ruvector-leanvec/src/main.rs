//! `leanvec-demo` — runnable A/B/C bench.
//!
//! Generates an anisotropic synthetic corpus, splits it into train / database
//! / query partitions, builds three indices and prints a comparison table of
//! per-vector memory footprint, recall@10 against brute-force ground truth,
//! and total wall time per 100 queries.
//!
//! No external bench harness; `cargo run --release -p ruvector-leanvec` is the
//! reproducible entry point and produces the numbers used in the research
//! document.

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use ruvector_leanvec::{
    FlatIndex, LeanVecIndex, LvqIndex, Neighbor, Projection, VectorIndex,
};

/// Default knobs. Override with `LEANVEC_N`, `LEANVEC_D`, `LEANVEC_Q`,
/// `LEANVEC_R`, `LEANVEC_RERANK`. Defaults pick a SIFT-like scale that runs
/// in seconds on a laptop while remaining honest about recall.
struct Cfg {
    n_db: usize,
    n_train: usize,
    n_query: usize,
    d: usize,
    r: usize,
    k: usize,
    rerank_mult: usize,
    seed: u64,
}

impl Cfg {
    fn from_env() -> Self {
        fn get<T: std::str::FromStr>(k: &str, default: T) -> T {
            std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
        }
        let d: usize = get("LEANVEC_D", 128);
        Self {
            n_db: get("LEANVEC_N", 10_000),
            n_train: get("LEANVEC_TRAIN", 2_000),
            n_query: get("LEANVEC_Q", 200),
            d,
            r: get("LEANVEC_R", d / 2),
            k: get("LEANVEC_K", 10),
            rerank_mult: get("LEANVEC_RERANK", 4),
            seed: get("LEANVEC_SEED", 17),
        }
    }
}

/// Anisotropic GMM-ish data: a handful of latent factors dominate the variance,
/// the rest is small noise. This is the regime LeanVec targets — PCA cleanly
/// recovers the signal subspace and most of the L2 is preserved.
fn synth_dataset(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let latent = (d / 8).max(4);
    let mut basis = vec![0.0_f32; latent * d];
    for k in 0..latent {
        for j in 0..d {
            basis[k * d + j] = rng.gen::<f32>() - 0.5;
        }
        let mut s = 0.0;
        for j in 0..d {
            s += basis[k * d + j] * basis[k * d + j];
        }
        let inv = 1.0 / s.sqrt();
        for j in 0..d {
            basis[k * d + j] *= inv;
        }
    }
    let mut out = vec![0.0_f32; n * d];
    for i in 0..n {
        let row = &mut out[i * d..(i + 1) * d];
        for k in 0..latent {
            let coef = (rng.gen::<f32>() - 0.5) * 6.0;
            for j in 0..d {
                row[j] += coef * basis[k * d + j];
            }
        }
        for j in 0..d {
            row[j] += (rng.gen::<f32>() - 0.5) * 0.05;
        }
    }
    out
}

fn recall_at_k(truth: &[Vec<Neighbor>], got: &[Vec<Neighbor>], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (t, g) in truth.iter().zip(got.iter()) {
        let truth_ids: std::collections::HashSet<u32> =
            t.iter().take(k).map(|n| n.id).collect();
        for n in g.iter().take(k) {
            if truth_ids.contains(&n.id) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f64 / total as f64
}

fn search_all(idx: &dyn VectorIndex, queries: &[f32], d: usize, k: usize) -> Vec<Vec<Neighbor>> {
    let nq = queries.len() / d;
    (0..nq)
        .map(|i| idx.search(&queries[i * d..(i + 1) * d], k))
        .collect()
}

fn main() {
    let cfg = Cfg::from_env();
    println!(
        "[cfg] n_db={} n_train={} n_query={} d={} r={} k={} rerank_mult={} seed={}",
        cfg.n_db, cfg.n_train, cfg.n_query, cfg.d, cfg.r, cfg.k, cfg.rerank_mult, cfg.seed
    );

    println!("[1/4] generating dataset…");
    let db = synth_dataset(cfg.n_db, cfg.d, cfg.seed);
    let queries = synth_dataset(cfg.n_query, cfg.d, cfg.seed.wrapping_add(101));
    // Train projection on the database prefix (LeanVec trains on a sample
    // drawn from the same distribution as the indexed vectors).
    let train_n = cfg.n_train.min(cfg.n_db);
    let train_slice = &db[..train_n * cfg.d];

    println!("[2/4] training PCA projection r={}…", cfg.r);
    let t = Instant::now();
    let proj = Projection::train_pca(train_slice, train_n, cfg.d, cfg.r, cfg.seed);
    println!("       trained in {:.2?}", t.elapsed());

    println!("[3/4] building indices…");
    let t = Instant::now();
    let mut flat = FlatIndex::new(cfg.d);
    for i in 0..cfg.n_db {
        flat.add(&db[i * cfg.d..(i + 1) * cfg.d]);
    }
    let build_flat = t.elapsed();

    let t = Instant::now();
    let mut lvq = LvqIndex::new(cfg.d);
    for i in 0..cfg.n_db {
        lvq.add(&db[i * cfg.d..(i + 1) * cfg.d]);
    }
    let build_lvq = t.elapsed();

    let t = Instant::now();
    let mut lv = LeanVecIndex::new(proj.clone(), cfg.rerank_mult);
    for i in 0..cfg.n_db {
        lv.add(&db[i * cfg.d..(i + 1) * cfg.d]);
    }
    let build_lv = t.elapsed();

    println!(
        "       build: flat {:.2?} | lvq {:.2?} | leanvec {:.2?}",
        build_flat, build_lvq, build_lv
    );

    println!("[4/4] querying…");
    let t = Instant::now();
    let truth = search_all(&flat, &queries, cfg.d, cfg.k);
    let qt_flat = t.elapsed();

    let t = Instant::now();
    let got_lvq = search_all(&lvq, &queries, cfg.d, cfg.k);
    let qt_lvq = t.elapsed();

    let t = Instant::now();
    let got_lv = search_all(&lv, &queries, cfg.d, cfg.k);
    let qt_lv = t.elapsed();

    let r_lvq = recall_at_k(&truth, &got_lvq, cfg.k);
    let r_lv = recall_at_k(&truth, &got_lv, cfg.k);

    let per_q = |t: std::time::Duration| t.as_nanos() as f64 / cfg.n_query as f64;

    println!();
    println!("=== ruvector-leanvec results ===");
    println!(
        "{:<14} {:>14} {:>12} {:>16} {:>12}",
        "variant", "bytes/vec", "recall@10", "ns/query", "rel-speed"
    );
    println!("{}", "-".repeat(72));
    let bv = |b: usize| (b as f64) / (cfg.n_db as f64);
    let ref_ns = per_q(qt_flat);
    let row = |name: &str, bytes: usize, recall: f64, dur: std::time::Duration| {
        let ns = per_q(dur);
        println!(
            "{:<14} {:>14.1} {:>12.4} {:>16.1} {:>11.2}x",
            name,
            bv(bytes),
            recall,
            ns,
            ref_ns / ns
        );
    };
    row("FlatF32", flat.bytes(), 1.0, qt_flat);
    row("LVQ-8", lvq.bytes(), r_lvq, qt_lvq);
    row("LeanVec-LVQ", lv.bytes(), r_lv, qt_lv);
    println!();
    println!(
        "(rel-speed > 1 means faster than FlatF32; LeanVec retains f32 originals for rerank)"
    );
}
