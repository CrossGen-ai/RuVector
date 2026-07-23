//! End-to-end benchmark: 3 variants × recall + latency + memory.
//! All numbers are real — no mocks, no fabricated results.
//!
//! Run: `cargo run --release -p ruvector-anisotropic-pq --example bench`

use std::time::Instant;

use ruvector_anisotropic_pq::{data, AnisotropicPq, LossKind, PqConfig};

fn main() {
    let n = std::env::var("BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(50_000);
    let d = 128usize;
    let m = 16usize;
    let k = 256usize;
    let nq = 200usize;
    let top_k = 100usize;

    println!("# ruvector-anisotropic-pq benchmark");
    println!("n={n} d={d} m={m} k={k} queries={nq} top_k={top_k}");
    println!("host: {}", std::env::consts::OS);

    print!("building synthetic dataset... ");
    let t0 = Instant::now();
    // Two datasets side-by-side: (i) L2-normalized (cosine/MIPS regime,
    // where ScaNN was designed), (ii) raw Gaussian mixture (varied norms).
    let (raw, _) = data::synthetic_gaussian(n, d, 32, 42);
    let data = data::l2_normalize(&raw, d);
    let queries = data::sample(&data, d, nq, 99);
    println!("{:.2}s", t0.elapsed().as_secs_f32());

    let variants = [
        ("baseline_pq        ", LossKind::Reconstruction),
        ("anisotropic_eta=4  ", LossKind::Anisotropic { eta: 4.0 }),
        ("anisotropic_eta=8  ", LossKind::Anisotropic { eta: 8.0 }),
        ("learned_norm_2..6  ", LossKind::LearnedNorm { eta_min: 2.0, eta_max: 6.0 }),
    ];

    println!();
    println!("{:22} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "variant", "train_s", "recall_L2", "recall_MIPS", "mse_par", "mse_perp", "qps", "bytes/vec");
    println!("{}", "-".repeat(110));

    for (name, loss) in variants {
        let cfg = PqConfig { d, m, k, iters: 15, loss, seed: 1234 };
        let t = Instant::now();
        let pq = AnisotropicPq::train(&cfg, &data);
        let train_s = t.elapsed().as_secs_f32();

        let t = Instant::now();
        let codes = pq.encode_batch(&data);
        let _enc_s = t.elapsed().as_secs_f32();

        use ruvector_anisotropic_pq::pq::Metric;
        let recall_l2   = ruvector_anisotropic_pq::pq::eval_recall_metric(&pq, &data, &codes, &queries, top_k, Metric::L2);
        let recall_mips = ruvector_anisotropic_pq::pq::eval_recall_metric(&pq, &data, &codes, &queries, top_k, Metric::Mips);
        let (mse_par, mse_perp) = ruvector_anisotropic_pq::pq::mse_decomposition(&pq, &data, &codes);

        // QPS: run queries with LUT + search, exclude LUT build negligible.
        let t = Instant::now();
        let mut sink = 0f32;
        for qi in 0..nq {
            let q = &queries[qi * d..(qi + 1) * d];
            let lut = pq.build_lut(q, true);
            let r = pq.search(&lut, &codes, top_k);
            sink += r[0].0;
        }
        let dt = t.elapsed().as_secs_f32();
        let qps = nq as f32 / dt;
        let _us = dt * 1e6 / nq as f32;
        let bpv = pq.bytes_per_vector();

        println!("{:22} | {:>8.2} | {:>10.3} | {:>11.3} | {:>10.5} | {:>10.5} | {:>10.1} | {:>10}",
            name, train_s, recall_l2, recall_mips, mse_par, mse_perp, qps, bpv);
        std::hint::black_box(sink);
    }

    println!();
    println!("full-float baseline: {} bytes/vec (f32 x {})", d * 4, d);
}
