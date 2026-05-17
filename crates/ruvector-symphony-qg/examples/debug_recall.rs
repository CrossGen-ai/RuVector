use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use ruvector_symphony_qg::{AnnIndex, FlatIndex, SymphonyQgIndex};

fn gauss(n: usize, d: usize, kc: usize, sigma: f32, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..kc).map(|_| (0..d).map(|_| rng.gen_range(-5.0..5.0)).collect()).collect();
    let normal = Normal::new(0.0, sigma).unwrap();
    let mut out = Vec::with_capacity(n * d);
    for i in 0..n {
        let c = &centers[i % kc];
        for j in 0..d { out.push(c[j] + normal.sample(&mut rng) as f32); }
    }
    out
}

fn main() {
    let d = 64; let n = 2_000; let k = 10;
    let data = gauss(n, d, 16, 1.0, 21);
    let qs = gauss(5, d, 16, 1.0, 22);
    let flat = FlatIndex::new(d, data.clone());
    let sym = SymphonyQgIndex::build(d, data, 16, 256, 10, 24, 128, 1.2, 7).with_ef_search(128);

    // Degree distribution.
    let degs: Vec<usize> = sym.adj.iter().map(|v| v.len()).collect();
    let avg_deg = degs.iter().sum::<usize>() as f32 / degs.len() as f32;
    let min_deg = *degs.iter().min().unwrap();
    let max_deg = *degs.iter().max().unwrap();
    println!("graph: avg_deg={avg_deg:.1} min={min_deg} max={max_deg}");

    for q_idx in 0..5 {
        let q = &qs[q_idx*d..(q_idx+1)*d];
        let truth = flat.search(q, k);
        let got = sym.search(q, k);
        println!("--- q{q_idx} ---");
        println!("truth ids: {:?}", truth.iter().map(|x| x.0).collect::<Vec<_>>());
        println!("got ids:   {:?}", got.iter().map(|x| x.0).collect::<Vec<_>>());
        println!("truth dists: {:?}", truth.iter().map(|x| x.1).collect::<Vec<_>>());
        println!("got dists(ADT): {:?}", got.iter().map(|x| x.1).collect::<Vec<_>>());
    }
}
