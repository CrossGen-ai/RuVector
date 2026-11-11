//! Benchmark: compare brute-force vs LeanVec-PCA vs random-projection
//! shortlist on a synthetic clustered float32 dataset.

use ruvector_leanvec_shortlist::{
    gen_clustered, recall_at_k, time_it, BruteForce, LeanVecShortlist, Projection, Searcher,
    XorShift64,
};

fn main() {
    let n = 20_000usize;
    let d = 128usize;
    let clusters = 64;
    let signal_dim = 20; // true low-rank dimensionality of the signal
    let k = 10usize;
    let n_queries = 200usize;
    let seed = 0xC0FFEE_u64;

    println!("# ruvector-leanvec-shortlist bench");
    println!(
        "n={} d={} clusters={} signal_dim={} k={} queries={}",
        n, d, clusters, signal_dim, k, n_queries
    );

    let ds = gen_clustered(n, d, clusters, signal_dim, seed);

    // Generate queries from the same distribution.
    let mut rng = XorShift64::new(seed ^ 0xDEAD_BEEF);
    let mut queries = vec![0.0f32; n_queries * d];
    for i in 0..n_queries {
        // Pick a random dataset point and perturb it, so the task is
        // nontrivial but has a real ground truth nearby.
        let src = (rng.next_u64() as usize) % n;
        let row = ds.row(src);
        for j in 0..d {
            queries[i * d + j] = row[j] + 0.3 * rng.normal();
        }
    }

    // Ground truth via brute force.
    let bf = BruteForce { ds: &ds };
    let mut gt: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_queries);
    let t_bf = time_it(|| {
        for i in 0..n_queries {
            let q = &queries[i * d..(i + 1) * d];
            gt.push(bf.search(q, k));
        }
    });

    // Learn PCA-16 and PCA-32 projections.
    let t_pca16 = std::time::Instant::now();
    let p16 = Projection::learn_pca(&ds, 16, 6, 4_000, seed);
    let t_pca16 = t_pca16.elapsed().as_secs_f64();

    let t_pca32 = std::time::Instant::now();
    let p32 = Projection::learn_pca(&ds, 32, 6, 4_000, seed);
    let t_pca32 = t_pca32.elapsed().as_secs_f64();

    // Random JL projection to 16-d for ablation.
    let rnd16 = Projection::random_gaussian(d, 16, seed);

    let shortlist = 200;

    let lv16 = LeanVecShortlist::build(&ds, p16, shortlist, "leanvec_pca16_L200");
    let lv32 = LeanVecShortlist::build(&ds, p32, shortlist, "leanvec_pca32_L200");
    let rv16 = LeanVecShortlist::build(&ds, rnd16, shortlist, "random_jl16_L200");

    let searchers: Vec<&dyn Searcher> = vec![&lv16, &lv32, &rv16];

    println!();
    println!("## Projection training time");
    println!("pca16_train_s   = {:.4}", t_pca16);
    println!("pca32_train_s   = {:.4}", t_pca32);

    println!();
    println!("## Query results (averaged over {} queries, k={})", n_queries, k);
    println!("brute_force_d    total_s={:.4}  per_q_ms={:.3}  recall@k=1.000",
        t_bf, 1000.0 * t_bf / n_queries as f64);

    for s in &searchers {
        let mut results = Vec::with_capacity(n_queries);
        let t = time_it(|| {
            for i in 0..n_queries {
                let q = &queries[i * d..(i + 1) * d];
                results.push(s.search(q, k));
            }
        });
        let mut avg_recall = 0.0f32;
        for i in 0..n_queries {
            avg_recall += recall_at_k(&results[i], &gt[i], k);
        }
        avg_recall /= n_queries as f32;
        let per_q_ms = 1000.0 * t / n_queries as f64;
        println!(
            "{:<16} total_s={:.4}  per_q_ms={:.3}  recall@k={:.3}  speedup_vs_bf={:.2}x",
            s.name(),
            t,
            per_q_ms,
            avg_recall,
            t_bf / t
        );
    }
}
