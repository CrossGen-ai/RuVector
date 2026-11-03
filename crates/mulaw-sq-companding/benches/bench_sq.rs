use mulaw_sq_companding::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

fn gauss(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let d = Normal::new(0.0f32, 1.0).unwrap();
    (0..n).map(|_| (0..dim).map(|_| d.sample(&mut rng)).collect()).collect()
}

fn bench_encode(name: &str, codec: &dyn ScalarCodec, data: &[Vec<f32>]) {
    let dim = data[0].len();
    let mut code = vec![0u8; dim];
    // warmup
    for v in data.iter().take(64) {
        codec.encode(v, &mut code);
    }
    let t0 = Instant::now();
    let iters = 5;
    for _ in 0..iters {
        for v in data {
            codec.encode(v, &mut code);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let total = (data.len() * iters) as f64;
    println!("{:16} encode: {:.3} Mvec/s", name, total / elapsed / 1e6);
}

fn bench_decode(name: &str, codec: &dyn ScalarCodec, data: &[Vec<f32>]) {
    let dim = data[0].len();
    let mut code = vec![0u8; dim];
    let mut out = vec![0f32; dim];
    codec.encode(&data[0], &mut code);
    let t0 = Instant::now();
    let iters = 50_000;
    for _ in 0..iters {
        codec.decode(&code, &mut out);
    }
    let elapsed = t0.elapsed().as_secs_f64();
    println!("{:16} decode: {:.3} Mvec/s", name, iters as f64 / elapsed / 1e6);
}

fn main() {
    let dim = 128;
    let base = gauss(10_000, dim, 11);
    let queries = gauss(500, dim, 22);

    let lin = LinearSq8::fit(&base, 0.999);
    let mu255 = MuLawSq8::fit(&base, 0.999, 255.0);
    let mu_learn = MuLawSq8::fit_learned_mu(&base, 0.999);

    println!("# μ-law SQ bench — dim={dim}, N={}, Q={}", base.len(), queries.len());
    println!("# learned mu = {:.2}", mu_learn.mu);

    for (name, codec) in [
        ("linear_sq8", &lin as &dyn ScalarCodec),
        ("mulaw_mu255", &mu255 as &dyn ScalarCodec),
        ("mulaw_learned", &mu_learn as &dyn ScalarCodec),
    ] {
        bench_encode(name, codec, &base);
        bench_decode(name, codec, &base);
        let mse = reconstruction_mse(codec, &base);
        let r10 = recall_at_k(codec, &base, &queries, 10);
        let r100 = recall_at_k(codec, &base, &queries, 100);
        println!(
            "{:16} mse={:.6e}  recall@10={:.4}  recall@100={:.4}",
            name, mse, r10, r100
        );
        println!();
    }

    // Memory footprint summary
    let fp_bytes = base.len() * dim * std::mem::size_of::<f32>();
    let sq_bytes = base.len() * dim; // 1 byte / dim
    println!(
        "# Memory: float32={:.2} MiB  sq8={:.2} MiB  compression={}x",
        fp_bytes as f64 / 1024.0 / 1024.0,
        sq_bytes as f64 / 1024.0 / 1024.0,
        fp_bytes / sq_bytes
    );
}
