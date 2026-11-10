//! PPR benchmark: compares FullL2, PQ-ADC, and PermutationPrefix rerankers.
//!
//! Data: Gaussian mixture of `n_clusters` clusters in `dim` dimensions.
//! The ANN shortlist is simulated by taking a random subset of size
//! `shortlist` from the dataset (noisy first-stage recall), and for each
//! query recall@k is measured against the brute-force f32 top-k over the
//! *full* dataset.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use ruvector_permutation_prefix::{
    l2::L2Reranker, pq::PqReranker, ppr::PermutationPrefixReranker, recall_at_k, sq_l2, Reranker,
};
use std::time::Instant;

fn gaussian_mixture(
    n: usize,
    dim: usize,
    n_clusters: usize,
    sep: f32,
    noise_std: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.gen_range(-sep..sep)).collect())
        .collect();
    let normal = Normal::new(0.0, noise_std as f64).unwrap();
    (0..n)
        .map(|i| {
            let c = &centers[i % n_clusters];
            c.iter()
                .map(|&x| x + normal.sample(&mut rng) as f32)
                .collect()
        })
        .collect()
}

fn brute_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (sq_l2(query, v), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

struct Measurement {
    name: &'static str,
    bytes_per_vector: usize,
    build_us: f64,
    latencies_us: Vec<f64>,
    recalls: Vec<f64>,
}

fn p(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx]
}

fn report(ms: &[Measurement], n_vectors: usize) {
    println!(
        "\n{:<20} {:>8} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "variant", "bpv", "build_us", "mean_us", "p50_us", "p95_us", "qps", "recall@k"
    );
    println!("{}", "-".repeat(100));
    for m in ms {
        let mut l = m.latencies_us.clone();
        l.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = m.latencies_us.iter().sum::<f64>() / m.latencies_us.len() as f64;
        let p50 = p(&l, 0.50);
        let p95 = p(&l, 0.95);
        let qps = if mean > 0.0 { 1_000_000.0 / mean } else { 0.0 };
        let recall = m.recalls.iter().sum::<f64>() / m.recalls.len() as f64;
        let total_bytes = m.bytes_per_vector * n_vectors;
        println!(
            "{:<20} {:>8} {:>12.1} {:>10.2} {:>10.2} {:>10.2} {:>10.0} {:>9.4}",
            m.name, m.bytes_per_vector, m.build_us, mean, p50, p95, qps, recall
        );
        println!("{:<20}   total_bytes={} MB", "", total_bytes / 1_048_576);
    }
}

fn main() {
    // Config.
    let n_vectors = std::env::var("PPR_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_000usize);
    let dim = std::env::var("PPR_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128usize);
    let n_clusters = 32usize;
    let n_queries = 200usize;
    let k = 10usize;
    let shortlist_size = 200usize; // simulated first-stage shortlist
    let seed = 2026;

    println!("PPR benchmark");
    println!(
        "  n={}  dim={}  clusters={}  queries={}  k={}  shortlist={}",
        n_vectors, dim, n_clusters, n_queries, k, shortlist_size
    );

    let data = gaussian_mixture(n_vectors, dim, n_clusters, 4.0, 1.0, seed);
    let queries = gaussian_mixture(n_queries, dim, n_clusters, 4.0, 1.0, seed + 1);

    // Build variants.
    let t0 = Instant::now();
    let l2 = L2Reranker::new(data.clone());
    let l2_build = t0.elapsed().as_micros() as f64;

    // Match roughly the same rerank-sidecar byte budget across variants:
    //   PQ-ADC   : 16 subs × 1B code  = 16 B/vec  (k=256 centroids per subspace)
    //   PPR      : 16 ranks × 2B      = 32 B/vec  (n_pivots=256, prefix_m=16)
    //   PPR-tiny : 8 ranks × 2B       = 16 B/vec  (n_pivots=128, prefix_m=8)
    let (pq, pq_build) = PqReranker::build(&data, 16, 256, 12, seed + 7);
    let (ppr, ppr_build) = PermutationPrefixReranker::build(&data, 256, 16, seed + 11);
    let (ppr_tiny, ppr_tiny_build) =
        PermutationPrefixReranker::build(&data, 128, 8, seed + 13);

    // Build first-stage shortlists: the top-k is anchored into a shortlist
    // padded with random distractors drawn uniformly from the dataset.
    // This is the typical "post-first-stage" rerank setting where the
    // first stage already filtered to a modest candidate pool but with
    // noise from HNSW traversal, LSH bucketing, or IVF probes.
    let mut rng = StdRng::seed_from_u64(seed + 101);
    let truths: Vec<Vec<u32>> = queries.iter().map(|q| brute_topk(&data, q, k)).collect();
    let shortlists: Vec<Vec<u32>> = queries
        .iter()
        .enumerate()
        .map(|(qi, _q)| {
            let mut set: Vec<u32> = truths[qi].clone();
            while set.len() < shortlist_size {
                let cand: u32 = rng.gen_range(0..data.len() as u32);
                if !set.contains(&cand) {
                    set.push(cand);
                }
            }
            set.shuffle(&mut rng);
            set
        })
        .collect();

    // Run each variant.
    let mut ms = Vec::new();
    for (name, bpv, build_us, boxed) in [
        (
            "FullL2",
            l2.bytes_per_vector(),
            l2_build,
            Box::new(l2) as Box<dyn Reranker>,
        ),
        (
            "PQ-ADC",
            pq.bytes_per_vector(),
            pq_build.as_micros() as f64,
            Box::new(pq),
        ),
        (
            "PermutationPrefix",
            ppr.bytes_per_vector(),
            ppr_build.as_micros() as f64,
            Box::new(ppr),
        ),
        (
            "PermutationPrefix-tiny",
            ppr_tiny.bytes_per_vector(),
            ppr_tiny_build.as_micros() as f64,
            Box::new(ppr_tiny),
        ),
    ] {
        let mut latencies = Vec::with_capacity(n_queries);
        let mut recalls = Vec::with_capacity(n_queries);
        for (qi, q) in queries.iter().enumerate() {
            let t = Instant::now();
            let top = boxed.rerank(q, &shortlists[qi], k);
            latencies.push(t.elapsed().as_nanos() as f64 / 1000.0);
            recalls.push(recall_at_k(&top, &truths[qi]));
        }
        ms.push(Measurement {
            name,
            bytes_per_vector: bpv,
            build_us,
            latencies_us: latencies,
            recalls,
        });
    }

    report(&ms, n_vectors);

    println!("\nnotes:");
    println!("  * FullL2 is the ground-truth reranker; its recall@k is 1.0 by definition.");
    println!("  * PQ-ADC            (n_sub=16, k=256, u8 codes)       → 16 bytes/vec");
    println!("  * PermutationPrefix (n_pivots=256, prefix_m=16, u16)  → 32 bytes/vec");
    println!("  * PPR-tiny          (n_pivots=128, prefix_m=8, u16)   → 16 bytes/vec");
    println!("  * Shortlist always contains the true top-k, so the ceiling recall is 1.0.");
    println!("  * bpv = bytes stored per vector (rerank sidecar; the first-stage index is separate).");
}
