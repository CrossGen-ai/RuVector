//! Real benchmark harness: synthetic Gaussian dataset, exact ground truth,
//! measure recall@k and cost across the three modes.

use crate::build::{build_hnsw_like, BuildParams};
use crate::graph::l2_sq;
use crate::search::{search, Mode};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub mode: &'static str,
    pub recall_at_k: f64,
    pub avg_dist_evals: f64,
    pub avg_nodes_visited: f64,
    pub qps: f64,
    pub build_seconds: f64,
    pub extra_bytes_per_node: usize,
}

pub struct BenchConfig {
    pub n: usize,
    pub dim: usize,
    pub n_queries: usize,
    pub k: usize,
    pub ef: usize,
    pub m_max: usize,
    pub ef_construction: usize,
    pub r_max: usize,
    pub n_clusters: usize,
    pub seed: u64,
}

pub fn default_config() -> BenchConfig {
    BenchConfig {
        n: 20_000,
        dim: 64,
        n_queries: 300,
        k: 10,
        ef: 16,
        m_max: 16,
        ef_construction: 100,
        r_max: 12,
        n_clusters: 32,
        seed: 42,
    }
}

pub fn gen_dataset(cfg: &BenchConfig) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    use rand_distr::{Distribution, Normal};
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    // Mixture-of-Gaussians dataset: `n_clusters` broad, overlapping components
    // in a `dim`-dimensional space. Cluster centers are spread modestly so
    // that inter-cluster routing is required but not impossibly hard.
    let center_std = 1.0f32;
    let centers: Vec<Vec<f32>> = (0..cfg.n_clusters)
        .map(|_| {
            let n = Normal::new(0.0f32, center_std).unwrap();
            (0..cfg.dim).map(|_| n.sample(&mut rng)).collect()
        })
        .collect();
    let intra_std = 0.5f32;
    let intra = Normal::new(0.0f32, intra_std).unwrap();
    let mut vecs = Vec::with_capacity(cfg.n);
    for _ in 0..cfg.n {
        let c = &centers[rng.gen_range(0..cfg.n_clusters)];
        let v: Vec<f32> = c.iter().map(|x| x + intra.sample(&mut rng)).collect();
        vecs.push(v);
    }
    let mut queries = Vec::with_capacity(cfg.n_queries);
    for _ in 0..cfg.n_queries {
        let c = &centers[rng.gen_range(0..cfg.n_clusters)];
        let v: Vec<f32> = c.iter().map(|x| x + intra.sample(&mut rng)).collect();
        queries.push(v);
    }
    (vecs, queries)
}

pub fn exact_topk(vectors: &[Vec<f32>], q: &[f32], k: usize) -> Vec<usize> {
    let mut ranked: Vec<(f32, usize)> =
        vectors.iter().enumerate().map(|(i, v)| (l2_sq(q, v), i)).collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    ranked.into_iter().take(k).map(|(_, i)| i).collect()
}

pub fn run_bench(cfg: &BenchConfig) -> Vec<BenchResult> {
    use std::time::Instant;
    let (vectors, queries) = gen_dataset(cfg);

    // Ground truth
    let truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| exact_topk(&vectors, q, cfg.k))
        .collect();

    // Baseline graph
    let t0 = Instant::now();
    let g_base = build_hnsw_like(
        &vectors,
        BuildParams { m_max: cfg.m_max, ef_construction: cfg.ef_construction },
    );
    let build_base = t0.elapsed().as_secs_f64();

    // Reverse-augmented (same graph, just add sidecar)
    let mut g_rev = g_base.clone();
    g_rev.build_reverse(cfg.r_max);
    let rev_extra_bytes = cfg.r_max * std::mem::size_of::<usize>();

    // Symmetrised graph (mutates forward adjacency; rebuild build to time it fairly)
    let t0 = Instant::now();
    let mut g_sym = build_hnsw_like(
        &vectors,
        BuildParams { m_max: cfg.m_max, ef_construction: cfg.ef_construction },
    );
    g_sym.symmetrize(&vectors);
    let build_sym = t0.elapsed().as_secs_f64();

    let mut results = Vec::new();

    for (mode, label, graph, build_s, extra) in [
        (Mode::Baseline, "baseline", &g_base, build_base, 0usize),
        (
            Mode::ExpandReverse,
            "expand-reverse",
            &g_rev,
            build_base, // reverse build is cheap; report base build
            rev_extra_bytes,
        ),
        (
            Mode::SymmetrizeBuild,
            "symmetrize-build",
            &g_sym,
            build_sym,
            0,
        ),
    ] {
        let mut hits = 0usize;
        let mut de = 0usize;
        let mut nv = 0usize;
        let t0 = Instant::now();
        for (qi, q) in queries.iter().enumerate() {
            let (top, s) = search(graph, &vectors, q, cfg.k, cfg.ef, mode);
            de += s.distance_evals;
            nv += s.nodes_visited;
            let gt = &truth[qi];
            for (_, idx) in &top {
                if gt.contains(idx) {
                    hits += 1;
                }
            }
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let recall = hits as f64 / (cfg.n_queries * cfg.k) as f64;
        results.push(BenchResult {
            mode: label,
            recall_at_k: recall,
            avg_dist_evals: de as f64 / cfg.n_queries as f64,
            avg_nodes_visited: nv as f64 / cfg.n_queries as f64,
            qps: cfg.n_queries as f64 / elapsed,
            build_seconds: build_s,
            extra_bytes_per_node: extra,
        });
    }
    results
}
