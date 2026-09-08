//! Benchmark harness: build one graph, wrap it in every backend, measure
//! memory + search latency + recall against brute-force ground truth.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::adjacency::{Adjacency, DeltaVarByteAdj, DenseAdj, ReorderedDeltaAdj};
use crate::graph::{build_graph, BuildParams};
use crate::reorder::bfs_permutation;
use crate::search::beam_search_multi;
use crate::{Distance, NodeId, SqEuclid, Vector};

#[derive(Clone, Copy)]
pub struct BenchConfig {
    pub n: usize,
    pub dim: usize,
    pub n_queries: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub k: usize,
    pub n_clusters: usize,
    pub clustered: bool,
    pub seed: u64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            n: 5_000,
            dim: 32,
            n_queries: 100,
            m: 16,
            ef_construction: 96,
            ef_search: 48,
            k: 10,
            n_clusters: 32,
            clustered: true,
            seed: 0x5EEDu64,
        }
    }
}

pub struct BackendResult {
    pub name: &'static str,
    pub bytes: usize,
    pub avg_ns_per_query: u128,
    pub avg_recall: f32,
    pub avg_dist_evals: f32,
}

pub struct Report {
    pub cfg_label: String,
    pub n: usize,
    pub dim: usize,
    pub avg_edges: f32,
    pub baseline_bytes: usize,
    pub variants: Vec<BackendResult>,
}

pub fn gen_corpus(cfg: &BenchConfig) -> (Vec<Vector>, Vec<Vector>) {
    let mut rng = rand_chacha_from_seed(cfg.seed);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let corpus: Vec<Vector> = if cfg.clustered {
        // Cluster centres spread across the space, intra-cluster spread
        // large enough that clusters overlap (avoids a disconnected
        // graph). This is the regime we want to exercise: soft clusters
        // giving small deltas after BFS reordering.
        let centres: Vec<Vector> = (0..cfg.n_clusters)
            .map(|_| (0..cfg.dim).map(|_| normal.sample(&mut rng) * 3.0).collect())
            .collect();
        (0..cfg.n)
            .map(|i| {
                let c = &centres[i % cfg.n_clusters];
                (0..cfg.dim)
                    .map(|d| c[d] + normal.sample(&mut rng) * 1.5)
                    .collect()
            })
            .collect()
    } else {
        (0..cfg.n)
            .map(|_| (0..cfg.dim).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    };
    // Sample queries as perturbations of held-in-corpus points so that
    // the ground-truth nearest neighbour is well-defined and reachable.
    let queries: Vec<Vector> = (0..cfg.n_queries)
        .map(|_| {
            let anchor = &corpus[rng.gen_range(0..corpus.len())];
            (0..cfg.dim)
                .map(|d| anchor[d] + normal.sample(&mut rng) * 0.5)
                .collect()
        })
        .collect();
    (corpus, queries)
}

// Small self-contained deterministic RNG (no extra dep beyond `rand`).
fn rand_chacha_from_seed(seed: u64) -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(seed)
}

fn ground_truth(corpus: &[Vector], q: &[f32], k: usize, dist: &SqEuclid) -> Vec<NodeId> {
    let mut d: Vec<(f32, NodeId)> = (0..corpus.len() as u32)
        .map(|i| (dist.dist(q, &corpus[i as usize]), i))
        .collect();
    d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    d.into_iter().take(k).map(|(_, i)| i).collect()
}

fn recall(topk: &[(f32, NodeId)], gt: &[NodeId]) -> f32 {
    let hit: usize = topk
        .iter()
        .filter(|(_, id)| gt.contains(id))
        .count();
    hit as f32 / gt.len() as f32
}

fn eval_backend<A: Adjacency>(
    name: &'static str,
    adj: &A,
    corpus: &[Vector],
    queries: &[Vector],
    ground: &[Vec<NodeId>],
    dist: &SqEuclid,
    entries: &[NodeId],
    ef: usize,
    k: usize,
) -> BackendResult {
    // Warmup.
    for q in queries.iter().take(5) {
        let _ = beam_search_multi(q, corpus, adj, dist, entries, ef, k);
    }
    let mut total_ns: u128 = 0;
    let mut total_recall: f32 = 0.0;
    let mut total_evals: usize = 0;
    for (i, q) in queries.iter().enumerate() {
        let t0 = Instant::now();
        let res = beam_search_multi(q, corpus, adj, dist, entries, ef, k);
        total_ns += t0.elapsed().as_nanos();
        total_recall += recall(&res.topk, &ground[i]);
        total_evals += res.dist_evals;
    }
    BackendResult {
        name,
        bytes: adj.bytes(),
        avg_ns_per_query: total_ns / queries.len() as u128,
        avg_recall: total_recall / queries.len() as f32,
        avg_dist_evals: total_evals as f32 / queries.len() as f32,
    }
}

pub fn run(cfg: BenchConfig) -> Report {
    let (corpus, queries) = gen_corpus(&cfg);
    let dist = SqEuclid;
    // Build the shared graph once.
    let lists = build_graph(
        &corpus,
        &dist,
        BuildParams { m: cfg.m, ef_construction: cfg.ef_construction },
    );
    let avg_edges = lists.iter().map(|l| l.len()).sum::<usize>() as f32
        / lists.len().max(1) as f32;

    // Ground truth.
    let ground: Vec<Vec<NodeId>> = queries
        .iter()
        .map(|q| ground_truth(&corpus, q, cfg.k, &dist))
        .collect();

    // Build three backends over the same lists.
    let dense = DenseAdj::from_lists(lists.clone());
    let delta = DeltaVarByteAdj::from_lists(lists.clone());
    let perm = bfs_permutation(&lists, 4);
    let reord = ReorderedDeltaAdj::from_lists_with_permutation(lists.clone(), perm);

    let baseline_bytes = dense.bytes();
    // Multi-entry: 8 evenly-spaced starting points cover the id space
    // regardless of graph long-range connectivity.
    let n_entries = 8.min(cfg.n);
    let stride = (cfg.n / n_entries).max(1);
    let entries: Vec<NodeId> = (0..n_entries).map(|i| (i * stride) as NodeId).collect();

    let mut variants = Vec::new();
    variants.push(eval_backend(
        dense.name(), &dense, &corpus, &queries, &ground, &dist, &entries, cfg.ef_search, cfg.k,
    ));
    variants.push(eval_backend(
        delta.name(), &delta, &corpus, &queries, &ground, &dist, &entries, cfg.ef_search, cfg.k,
    ));
    variants.push(eval_backend(
        reord.name(), &reord, &corpus, &queries, &ground, &dist, &entries, cfg.ef_search, cfg.k,
    ));

    Report {
        cfg_label: format!(
            "n={} dim={} m={} ef_c={} ef_s={} k={} {}",
            cfg.n,
            cfg.dim,
            cfg.m,
            cfg.ef_construction,
            cfg.ef_search,
            cfg.k,
            if cfg.clustered { "clustered" } else { "isotropic" }
        ),
        n: cfg.n,
        dim: cfg.dim,
        avg_edges,
        baseline_bytes,
        variants,
    }
}

pub fn format_report(r: &Report) -> String {
    let mut s = String::new();
    s.push_str(&format!("cfg: {}\n", r.cfg_label));
    s.push_str(&format!(
        "graph: n={} dim={} avg_edges={:.2}\n",
        r.n, r.dim, r.avg_edges
    ));
    s.push_str(&format!(
        "| backend | bytes | vs baseline | ns/query | recall@k | evals/query |\n"
    ));
    s.push_str("|---|---:|---:|---:|---:|---:|\n");
    for v in &r.variants {
        let ratio = v.bytes as f32 / r.baseline_bytes as f32;
        s.push_str(&format!(
            "| {} | {} | {:.2}× | {} | {:.3} | {:.1} |\n",
            v.name, v.bytes, ratio, v.avg_ns_per_query, v.avg_recall, v.avg_dist_evals
        ));
    }
    s
}
