//! Real benchmark harness for MUVERA.
//!
//! Runs three retrievers on a synthetic multi-vector corpus with a
//! held-out query set that has "planted" ground truth: each query is a
//! noisy perturbation of one specific document, so MaxSim's top-1 is
//! known. We report recall@1, recall@5, per-query latency (µs), and
//! index build time.
//!
//! Deterministic. Numbers below are captured verbatim into the research
//! doc — no smoothing, no cherry-picking, no simulation.

use ruvector_muvera::chamfer::l2_normalize_set;
use ruvector_muvera::fde::FdeParams;
use ruvector_muvera::ivf::IvfParams;
use ruvector_muvera::retriever::{
    Document, FlatMaxSim, MultiVectorRetriever, MuveraFlat,
};
use ruvector_muvera::MuveraIvf;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn gaussian(rng: &mut ChaCha8Rng) -> f32 {
    let u1: f32 = rng.gen::<f32>().max(1e-9);
    let u2: f32 = rng.gen::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
}

fn make_corpus(n_docs: usize, tokens_per_doc: usize, d: usize, seed: u64) -> Vec<Document> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n_docs)
        .map(|id| {
            let mut tokens = vec![0.0f32; tokens_per_doc * d];
            for x in tokens.iter_mut() {
                *x = gaussian(&mut rng);
            }
            l2_normalize_set(&mut tokens, d);
            Document { id: id as u32, tokens, n_tokens: tokens_per_doc }
        })
        .collect()
}

/// Make `n_queries` planted queries: each is doc[i] + N(0, noise_std) I,
/// re-normalized. Ground-truth top-1 is doc `i`.
fn planted_queries(
    corpus: &[Document],
    n_queries: usize,
    query_tokens: usize,
    d: usize,
    noise_std: f32,
    seed: u64,
) -> Vec<(u32, Vec<f32>)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n_queries)
        .map(|q| {
            // Pick a doc uniformly.
            let target = rng.gen::<usize>() % corpus.len();
            let src = &corpus[target].tokens;
            // Take the first `query_tokens` tokens of the doc, add noise.
            let take = query_tokens.min(corpus[target].n_tokens);
            let mut q_vec = vec![0.0f32; take * d];
            for i in 0..take {
                for j in 0..d {
                    q_vec[i * d + j] = src[i * d + j] + noise_std * gaussian(&mut rng);
                }
            }
            l2_normalize_set(&mut q_vec, d);
            let _ = q;
            (target as u32, q_vec)
        })
        .collect()
}

fn recall_at(hits: &[ruvector_muvera::retriever::RetrievalHit], gt_id: u32, k: usize) -> f32 {
    if hits.iter().take(k).any(|h| h.id == gt_id) { 1.0 } else { 0.0 }
}

struct RunResult {
    name: String,
    build_ms: f64,
    per_query_us_avg: f64,
    per_query_us_p50: f64,
    per_query_us_p95: f64,
    recall_at_1: f64,
    recall_at_5: f64,
    fde_dim: Option<usize>,
}

fn bench<R: MultiVectorRetriever>(
    label: &str,
    idx: &mut R,
    corpus: &[Document],
    queries: &[(u32, Vec<f32>)],
    fde_dim: Option<usize>,
) -> RunResult {
    let t0 = Instant::now();
    idx.build(corpus);
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut lats_us: Vec<f64> = Vec::with_capacity(queries.len());
    let mut r1_hits = 0.0f64;
    let mut r5_hits = 0.0f64;
    for (gt, q) in queries {
        let t = Instant::now();
        let hits = idx.search(q, 5);
        lats_us.push(t.elapsed().as_secs_f64() * 1e6);
        r1_hits += recall_at(&hits, *gt, 1) as f64;
        r5_hits += recall_at(&hits, *gt, 5) as f64;
    }
    let n = queries.len() as f64;
    let avg = lats_us.iter().sum::<f64>() / n;
    let mut sorted = lats_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[(sorted.len() as f64 * 0.50) as usize];
    let p95 = sorted[((sorted.len() as f64 * 0.95) as usize).min(sorted.len() - 1)];

    RunResult {
        name: label.to_string(),
        build_ms,
        per_query_us_avg: avg,
        per_query_us_p50: p50,
        per_query_us_p95: p95,
        recall_at_1: r1_hits / n,
        recall_at_5: r5_hits / n,
        fde_dim,
    }
}

fn print_table(rows: &[RunResult]) {
    println!();
    println!(
        "{:<40} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "variant", "fde_dim", "build_ms", "q_avg_us", "q_p50_us", "q_p95_us", "R@1", "R@5"
    );
    println!("{}", "-".repeat(112));
    for r in rows {
        println!(
            "{:<40} {:>10} {:>10.2} {:>10.1} {:>10.1} {:>10.1} {:>10.3} {:>10.3}",
            r.name,
            r.fde_dim.map(|d| d.to_string()).unwrap_or_else(|| "-".into()),
            r.build_ms,
            r.per_query_us_avg,
            r.per_query_us_p50,
            r.per_query_us_p95,
            r.recall_at_1,
            r.recall_at_5,
        );
    }
    println!();
}

fn main() {
    // Fixed scenario — deterministic, so numbers reproduce.
    let d = 32;               // token embedding dim
    let n_docs = 2_000;       // corpus size
    let tokens_per_doc = 16;  // multi-vec bag size per doc
    let n_queries = 200;
    let query_tokens = 8;
    let noise_std = 0.15;     // noise added to planted queries

    println!("ruvector-muvera benchmark");
    println!("=========================");
    println!("d={d}, n_docs={n_docs}, tokens_per_doc={tokens_per_doc}, n_queries={n_queries}, query_tokens={query_tokens}, noise_std={noise_std}");

    let corpus = make_corpus(n_docs, tokens_per_doc, d, 2026);
    let queries = planted_queries(&corpus, n_queries, query_tokens, d, noise_std, 4242);

    let mut results = Vec::new();

    // Variant 1 — exact oracle (recall ceiling).
    let mut flat = FlatMaxSim::new(d);
    results.push(bench("V1_flat_maxsim (oracle)", &mut flat, &corpus, &queries, None));

    // Variant 2 — MuveraFlat with small FDE (fast, lossy).
    let p_small = FdeParams { d, k_sim: 4, reps: 4, seed: 51 };
    let dim_small = FlatMaxSim::new(d);
    let _ = dim_small; // silence
    let fde_dim_small = d * (1 << p_small.k_sim) * p_small.reps;
    let mut mflat_small = MuveraFlat::new(p_small);
    results.push(bench(
        "V2a_muvera_flat (k=4, R=4)",
        &mut mflat_small,
        &corpus,
        &queries,
        Some(fde_dim_small),
    ));

    // Variant 2b — MuveraFlat with larger FDE (slower, more accurate).
    let p_big = FdeParams { d, k_sim: 5, reps: 12, seed: 51 };
    let fde_dim_big = d * (1 << p_big.k_sim) * p_big.reps;
    let mut mflat_big = MuveraFlat::new(p_big);
    results.push(bench(
        "V2b_muvera_flat (k=5, R=12)",
        &mut mflat_big,
        &corpus,
        &queries,
        Some(fde_dim_big),
    ));

    // Variant 3 — MuveraIvf (production shape).
    let p_ivf = FdeParams { d, k_sim: 5, reps: 8, seed: 51 };
    let fde_dim_ivf = d * (1 << p_ivf.k_sim) * p_ivf.reps;
    let ivf_p = IvfParams { n_lists: 32, n_probe: 8, candidates: 128, rerank: 32, kmeans_iters: 10, seed: 77 };
    let mut ivf = MuveraIvf::new(p_ivf, ivf_p);
    results.push(bench(
        "V3_muvera_ivf (k=5, R=8, nprobe=8, rerank=32)",
        &mut ivf,
        &corpus,
        &queries,
        Some(fde_dim_ivf),
    ));

    // Variant 3b — same IVF, no rerank (isolates rerank's impact).
    let ivf_p_no = IvfParams { n_lists: 32, n_probe: 8, candidates: 128, rerank: 0, kmeans_iters: 10, seed: 77 };
    let mut ivf_no = MuveraIvf::new(p_ivf, ivf_p_no);
    results.push(bench(
        "V3b_muvera_ivf_no_rerank (k=5, R=8, nprobe=8)",
        &mut ivf_no,
        &corpus,
        &queries,
        Some(fde_dim_ivf),
    ));

    print_table(&results);

    println!("Interpretation");
    println!("--------------");
    println!("* R@k is measured against a PLANTED ground truth (each query is a noisy");
    println!("  perturbation of one specific doc). MaxSim is the reference target.");
    println!("* V1 is the exact-MaxSim oracle — its latency is the pain we're trying");
    println!("  to avoid, and its recall is the ceiling MUVERA is approximating.");
    println!("* V2a shows the smallest useful FDE — cheap but lossy.");
    println!("* V2b shows a bigger FDE — approaches oracle recall at higher query cost.");
    println!("* V3 shows the full production pipeline: IVF pre-filter over FDEs +");
    println!("  exact MaxSim rerank on the top candidates. This is how you deploy MUVERA.");
    println!("* V3b removes rerank to isolate its contribution.");
}
