use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use ruvector_bcps::{
    bench_once, binary::BinarySignatureSelector, exact::ExactL2Selector, recall_at_n,
};

fn gauss_unit(nlist: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..nlist)
        .map(|_| {
            let mut v: Vec<f32> = (0..d).map(|_| StandardNormal.sample(&mut rng)).collect();
            let n: f32 = v.iter().map(|x: &f32| x * x).sum::<f32>().sqrt();
            if n > 0.0 { for x in &mut v { *x /= n; } }
            v
        })
        .collect()
}

fn main() {
    println!("ruvector-bcps demo (unit-sphere Gaussian centroids & queries)");
    println!("=============================================================");

    for (nlist, d, nq, nprobe) in &[
        (4096usize, 64usize, 200usize, 16usize),
        (16384, 128, 200, 16),
        (65536, 128, 100, 32),
    ] {
        let centroids = gauss_unit(*nlist, *d, 1);
        let queries = gauss_unit(*nq, *d, 2);

        let exact = ExactL2Selector::new(centroids.clone());
        let exact_m = bench_once(&exact, &queries, *nprobe);

        println!(
            "\n== nlist={} d={} nprobe={} exact_us/q={:.2} ==",
            nlist, d, nprobe, exact_m.elapsed_ns as f64 / 1000.0
        );
        println!(
            "{:>6} {:>6} {:>8} {:>10} {:>10} {:>9} {:>8}",
            "rots", "shortM", "sigBytes", "exact_us", "bcps_us", "speedup", "recall"
        );

        for &rots in &[1usize, 2, 4] {
            for &m_mult in &[4, 8, 16] {
                let m = nprobe * m_mult;
                let bin = BinarySignatureSelector::with_rotations(
                    centroids.clone(), m, 42, rots,
                );
                let r = recall_at_n(&bin, &exact, &queries, *nprobe);
                let bm = bench_once(&bin, &queries, *nprobe);
                let speed = exact_m.elapsed_ns as f64 / bm.elapsed_ns.max(1) as f64;
                println!(
                    "{:>6} {:>6} {:>8} {:>10.2} {:>10.2} {:>8.2}x {:>7.3}",
                    rots,
                    m,
                    bin.bytes_per_centroid(),
                    exact_m.elapsed_ns as f64 / 1000.0,
                    bm.elapsed_ns as f64 / 1000.0,
                    speed,
                    r,
                );
            }
        }
    }
}
