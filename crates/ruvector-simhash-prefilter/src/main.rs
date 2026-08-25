//! Demo binary: generates a synthetic Gaussian corpus and measures
//! recall@10 + wall-clock latency for exact scan vs. 64/128/256-bit
//! SimHash prefilter cascades at multiple candidate multipliers.
//!
//! Emits a JSON summary on the last line so this can be piped into
//! the research doc:
//!
//!     cargo run --release -p ruvector-simhash-prefilter -- 20000 128 200 10

use rand::{rngs::StdRng, RngCore, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_simhash_prefilter::{
    recall_at_k, FlatPrefilterIndex, SketchFamily, SrpFamily,
};
use serde::Serialize;
use std::time::Instant;

#[derive(Serialize)]
struct Variant {
    label: String,
    bits: usize,
    candidate_mult: usize,
    recall_at_10: f32,
    p50_query_us: f64,
    mean_query_us: f64,
    sketch_bytes_per_vec: usize,
    total_sketch_bytes: usize,
}

#[derive(Serialize)]
struct Report {
    n_vectors: usize,
    dim: usize,
    n_queries: usize,
    k: usize,
    raw_footprint_bytes: usize,
    baseline_exact: Variant,
    prefilter_variants: Vec<Variant>,
}

/// Mixture-of-Gaussians corpus: `n_clusters` centers drawn from N(0, I),
/// then per-point noise N(0, sigma²·I). Clustered data has the angular
/// structure that SimHash exploits — pure isotropic noise is the worst
/// case for any angular hash. Real learned embeddings (BERT/CLIP/…) sit
/// closer to this regime than to isotropic noise.
fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    let noise = Normal::new(0.0f32, 0.35).unwrap();
    let n_clusters = 64usize;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng)).collect())
        .collect();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            (0..dim).map(|j| c[j] + noise.sample(&mut rng)).collect()
        })
        .collect()
}

fn time_us(mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_nanos() as f64 / 1000.0
}

fn percentile(mut xs: Vec<f64>, p: f64) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((xs.len() as f64 - 1.0) * p).round() as usize;
    xs[idx]
}

fn eval_exact(
    idx: &FlatPrefilterIndex<SrpFamily<1>, 1>,
    queries: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<(usize, f32)>>, Vec<f64>) {
    let mut truths = Vec::with_capacity(queries.len());
    let mut lat = Vec::with_capacity(queries.len());
    for q in queries {
        let mut r = Vec::new();
        let t = time_us(|| {
            r = idx.search_exact(q, k);
        });
        truths.push(r);
        lat.push(t);
    }
    (truths, lat)
}

macro_rules! run_variant {
    ($W:literal, $bits:expr, $vectors:expr, $queries:expr, $truths:expr, $k:expr, $mult:expr, $seed:expr) => {{
        let family = SrpFamily::<$W>::new($vectors[0].len(), $seed);
        let idx = FlatPrefilterIndex::build(family, $vectors.clone()).unwrap();
        let mut recalls = Vec::new();
        let mut lat = Vec::new();
        for (q, truth) in $queries.iter().zip($truths.iter()) {
            let mut res = Vec::new();
            let t = time_us(|| {
                res = idx.search_prefilter(q, $k, $mult).unwrap();
            });
            lat.push(t);
            recalls.push(recall_at_k(&res, truth));
        }
        let recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
        let mean = lat.iter().sum::<f64>() / lat.len() as f64;
        let p50 = percentile(lat.clone(), 0.5);
        Variant {
            label: format!("{}bit_mult{}", $bits, $mult),
            bits: $bits,
            candidate_mult: $mult,
            recall_at_10: recall,
            p50_query_us: p50,
            mean_query_us: mean,
            sketch_bytes_per_vec: <SrpFamily<$W> as SketchFamily<$W>>::sketch_bytes(),
            total_sketch_bytes: idx.sketch_footprint_bytes(),
        }
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(20_000);
    let dim: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(128);
    let n_q: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(200);
    let k: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(10);

    eprintln!(
        "[simhash-prefilter-demo] n={n} dim={dim} n_queries={n_q} k={k}"
    );

    let vectors = synth(n, dim, 0xA11CE);
    // Realistic queries: pick a random corpus point and perturb it. This
    // mirrors ANN benchmark practice (SIFT/GIST/deep1B query sets are drawn
    // from the same distribution as the base set) and produces queries whose
    // true NN is a well-defined cluster neighbour rather than a random point.
    let queries: Vec<Vec<f32>> = {
        let mut qrng = StdRng::seed_from_u64(0xB0B);
        let qnoise = Normal::new(0.0f32, 0.20).unwrap();
        (0..n_q)
            .map(|_| {
                let idx = (qrng.next_u64() as usize) % vectors.len();
                let base = &vectors[idx];
                base.iter().map(|x| x + qnoise.sample(&mut qrng)).collect()
            })
            .collect()
    };

    // Baseline exact — the "no prefilter" reference. Family width is irrelevant
    // for search_exact (it does not touch sketches), so use W=1 to keep the
    // matrix small.
    let exact_family = SrpFamily::<1>::new(dim, 1);
    let exact_idx = FlatPrefilterIndex::build(exact_family, vectors.clone()).unwrap();
    let (truths, exact_lat) = eval_exact(&exact_idx, &queries, k);
    let baseline_exact = Variant {
        label: "exact_bruteforce".to_string(),
        bits: 0,
        candidate_mult: 0,
        recall_at_10: 1.0,
        p50_query_us: percentile(exact_lat.clone(), 0.5),
        mean_query_us: exact_lat.iter().sum::<f64>() / exact_lat.len() as f64,
        sketch_bytes_per_vec: 0,
        total_sketch_bytes: 0,
    };

    let raw_footprint_bytes = exact_idx.raw_footprint_bytes();

    let mut variants = Vec::new();
    // 3 measured variants at recall-tuned candidate multipliers.
    // Sweep 1: fixed candidate pool (mult=40 → 400/20k = 2% of corpus scanned
    // exactly), scan bit-widths to isolate the effect of sketch fidelity.
    variants.push(run_variant!(1, 64, vectors, queries, truths, k, 40, 42));
    variants.push(run_variant!(2, 128, vectors, queries, truths, k, 40, 42));
    variants.push(run_variant!(4, 256, vectors, queries, truths, k, 40, 42));
    // Sweep 2: fixed sketch width (128-bit), vary candidate pool. Shows
    // the recall/latency knob applications actually tune.
    variants.push(run_variant!(2, 128, vectors, queries, truths, k, 10, 42));
    variants.push(run_variant!(2, 128, vectors, queries, truths, k, 20, 42));
    variants.push(run_variant!(2, 128, vectors, queries, truths, k, 80, 42));

    let report = Report {
        n_vectors: n,
        dim,
        n_queries: n_q,
        k,
        raw_footprint_bytes,
        baseline_exact,
        prefilter_variants: variants,
    };

    // Human-readable table
    eprintln!(
        "\n{:<24} {:>10} {:>10} {:>12} {:>14}",
        "variant", "recall@10", "p50 (us)", "mean (us)", "sketch bytes"
    );
    eprintln!("{}", "-".repeat(74));
    let print_row = |v: &Variant| {
        eprintln!(
            "{:<24} {:>10.4} {:>10.2} {:>12.2} {:>14}",
            v.label,
            v.recall_at_10,
            v.p50_query_us,
            v.mean_query_us,
            v.total_sketch_bytes
        );
    };
    print_row(&report.baseline_exact);
    for v in &report.prefilter_variants {
        print_row(v);
    }
    eprintln!(
        "\nraw_footprint = {} MB",
        report.raw_footprint_bytes / (1024 * 1024)
    );

    // Machine-readable last line
    println!("{}", serde_json::to_string(&report).unwrap());
}
