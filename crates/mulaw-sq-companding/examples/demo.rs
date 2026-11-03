use mulaw_sq_companding::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn gauss(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let d = Normal::new(0.0f32, 1.0).unwrap();
    (0..n).map(|_| (0..dim).map(|_| d.sample(&mut rng)).collect()).collect()
}

fn main() {
    let dim = 128;
    let base = gauss(2048, dim, 11);
    let queries = gauss(200, dim, 22);

    let lin = LinearSq8::fit(&base, 0.999);
    let mu255 = MuLawSq8::fit(&base, 0.999, 255.0);
    let mu_learn = MuLawSq8::fit_learned_mu(&base, 0.999);

    for codec in [&lin as &dyn ScalarCodec, &mu255 as &dyn ScalarCodec, &mu_learn as &dyn ScalarCodec] {
        let mse = reconstruction_mse(codec, &base);
        let r = recall_at_k(codec, &base, &queries, 10);
        println!("{:16} mse={:.6e}  recall@10={:.4}", codec.name(), mse, r);
    }
    println!("mu(learned) = {:.1}", mu_learn.mu);
}
