//! Demo: show MSE and recall@10 for UniformSQ4, HadamardSQ4, HadamardSQ8 on
//! a skewed synthetic dataset. Prints a table with real cargo-run numbers.

use ruvector_hadamard_sq::distance::{brute_force_recall, make_skewed_dataset};
use ruvector_hadamard_sq::quant::{Codec, HadamardSQ4, HadamardSQ8, UniformSQ4};

fn mse_asymmetric<C: Codec>(codec: &C, data: &[f32], dim: usize) -> f64 {
    let n = data.len() / dim;
    let stride = codec.code_bytes();
    let mut code = vec![0u8; stride];
    let mut total = 0.0f64;
    for i in 0..n {
        let x = &data[i * dim..(i + 1) * dim];
        codec.encode(x, &mut code);
        total += codec.l2_sq_asymmetric(x, &code) as f64;
    }
    total / (n as f64 * dim as f64)
}

fn main() {
    let dim = 64;
    let n = 20_000;
    let nq = 200;
    let k = 10;
    let data = make_skewed_dataset(n, dim, 42);
    let queries = make_skewed_dataset(nq, dim, 1337);

    println!("ruvector-hadamard-sq demo");
    println!("  dataset: skewed-synthetic, n={} nq={} d={} k={}", n, nq, dim, k);
    println!();

    let base = UniformSQ4::train(dim, &data);
    let hada4 = HadamardSQ4::train(dim, &data, 7);
    let hada8 = HadamardSQ8::train(dim, &data, 7);

    let bytes_f32 = (n * dim * 4) as f64;
    let bytes_base = (n * base.code_bytes()) as f64;
    let bytes_h4 = (n * hada4.code_bytes()) as f64;
    let bytes_h8 = (n * hada8.code_bytes()) as f64;

    println!(
        "{:<16} {:>12} {:>10} {:>12} {:>12}",
        "codec", "mse(resid)", "recall@10", "bytes", "compression"
    );
    println!("{:-<66}", "");

    let (bm, br) = (
        mse_asymmetric(&base, &data, dim),
        brute_force_recall(&base, &data, &queries, dim, k),
    );
    println!(
        "{:<16} {:>12.5} {:>10.3} {:>12} {:>11.2}x",
        base.name(),
        bm,
        br,
        bytes_base as u64,
        bytes_f32 / bytes_base
    );

    let (h4m, h4r) = (
        mse_asymmetric(&hada4, &data, dim),
        brute_force_recall(&hada4, &data, &queries, dim, k),
    );
    println!(
        "{:<16} {:>12.5} {:>10.3} {:>12} {:>11.2}x",
        hada4.name(),
        h4m,
        h4r,
        bytes_h4 as u64,
        bytes_f32 / bytes_h4
    );

    let (h8m, h8r) = (
        mse_asymmetric(&hada8, &data, dim),
        brute_force_recall(&hada8, &data, &queries, dim, k),
    );
    println!(
        "{:<16} {:>12.5} {:>10.3} {:>12} {:>11.2}x",
        hada8.name(),
        h8m,
        h8r,
        bytes_h8 as u64,
        bytes_f32 / bytes_h8
    );

    println!();
    println!("summary (vs uniform-sq4 baseline):");
    println!("  hadamard-sq4 MSE : {:.3}x", h4m / bm);
    println!("  hadamard-sq4 recall gain: +{:.3}", h4r - br);
    println!("  hadamard-sq8 MSE : {:.3}x", h8m / bm);
    println!("  hadamard-sq8 recall gain: +{:.3}", h8r - br);
}
