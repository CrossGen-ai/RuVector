use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_ivf_ball_pruning::*;

fn gen(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect()).collect()
}

fn kmeans(points: &[Vec<f32>], k: usize, iters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let dim = points[0].len();
    let mut c: Vec<Vec<f32>> = (0..k).map(|_| points[rng.gen_range(0..points.len())].clone()).collect();
    for _ in 0..iters {
        let mut sums = vec![vec![0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for p in points {
            let mut best = 0; let mut bd = f32::INFINITY;
            for (ci, cc) in c.iter().enumerate() {
                let d = l2_sq(p, cc);
                if d < bd { bd = d; best = ci; }
            }
            counts[best] += 1;
            for d in 0..dim { sums[best][d] += p[d]; }
        }
        for ci in 0..k {
            if counts[ci] > 0 {
                for d in 0..dim { c[ci][d] = sums[ci][d] / counts[ci] as f32; }
            }
        }
    }
    c
}

fn main() {
    let n = 10_000;
    let dim = 64;
    let n_clusters = 128;
    let n_queries = 200;
    let k = 10;

    println!("# IVF Ball Pruning demo");
    println!("# n={} dim={} clusters={} k={} queries={}", n, dim, n_clusters, k, n_queries);

    let data = gen(n, dim, 42);
    let queries = gen(n_queries, dim, 43);
    let centroids = kmeans(&data, n_clusters, 15, 44);
    let idx = IvfIndex::build(data.clone(), centroids);

    println!("# mean cluster size: {:.1}", idx.mean_cluster_size());
    println!("# mean r_max:        {:.4}",
        idx.r_max.iter().sum::<f32>() / idx.r_max.len() as f32);

    for &nprobe in &[8usize, 16, 32, 64] {
        let scanners: Vec<Box<dyn IvfScan>> = vec![
            Box::new(BaselineScan),
            Box::new(PartialAbandonScan),
            Box::new(BallPruneScan),
        ];
        for s in &scanners {
            let t0 = std::time::Instant::now();
            let mut agg = ScanStats::zero();
            let mut rec_sum = 0f32;
            for q in &queries {
                let (res, stats) = s.search(&idx, q, nprobe, k);
                agg.probes_scheduled += stats.probes_scheduled;
                agg.probes_actually_scanned += stats.probes_actually_scanned;
                agg.vectors_touched += stats.vectors_touched;
                agg.full_distances += stats.full_distances;
                let truth = exact_topk(&idx.vectors, q, k);
                rec_sum += recall(&res, &truth);
            }
            let elapsed = t0.elapsed();
            let qps = n_queries as f64 / elapsed.as_secs_f64();
            let rec = rec_sum / n_queries as f32;
            println!(
                "nprobe={:3} scan={:<16} qps={:>8.1} rec@{}={:.4} probes={:>5} scanned={:>5} touched={:>7} full={:>7}",
                nprobe, s.name(), qps, k, rec,
                agg.probes_scheduled / n_queries,
                agg.probes_actually_scanned / n_queries,
                agg.vectors_touched / n_queries,
                agg.full_distances / n_queries,
            );
        }
    }
}
