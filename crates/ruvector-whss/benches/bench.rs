use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use ruvector_whss::*;
use std::time::Instant;

fn gen_corpus(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()).collect()
}

fn gen_queries(nq: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..nq).map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()).collect()
}

fn main() {
    let n = std::env::var("WHSS_N").ok().and_then(|s| s.parse().ok()).unwrap_or(20000usize);
    let d = std::env::var("WHSS_D").ok().and_then(|s| s.parse().ok()).unwrap_or(128usize);
    let nq = std::env::var("WHSS_NQ").ok().and_then(|s| s.parse().ok()).unwrap_or(100usize);
    let k = 10usize;

    println!("# WHSS bench — n={} d={} nq={} k={}", n, d, nq, k);
    let corpus = gen_corpus(n, d, 1);
    let queries = gen_queries(nq, d, 2);

    // Ground truth via exact scan + its latency.
    let t0 = Instant::now();
    let mut truths = Vec::with_capacity(nq);
    for q in &queries {
        truths.push(exact_topk(&corpus, q, k));
    }
    let exact_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let exact_qps = nq as f64 / (exact_ms / 1000.0);
    println!("baseline/full_scan  : {:>8.2} ms total | {:>8.1} QPS | recall@10 = 1.0000", exact_ms, exact_qps);

    for &m_bits in &[128usize, 256, 512] {
        for &cand in &[64usize, 256, 1024] {
            // WHSS build
            let whss = Whss::new(d, m_bits, 7);
            let tb = Instant::now();
            let wsk = encode_corpus(&whss, &corpus);
            let whss_build_ms = tb.elapsed().as_secs_f64() * 1000.0;
            // SRP build
            let srp = Srp::new(d, m_bits, 7);
            let tbs = Instant::now();
            let ssk = srp_encode_corpus(&srp, &corpus);
            let srp_build_ms = tbs.elapsed().as_secs_f64() * 1000.0;

            // WHSS search
            let tw = Instant::now();
            let mut whss_recall = 0.0f32;
            for (i, q) in queries.iter().enumerate() {
                let pred = whss_search(&whss, &corpus, &wsk, q, k, cand);
                whss_recall += recall_at_k(&truths[i], &pred);
            }
            let whss_ms = tw.elapsed().as_secs_f64() * 1000.0;
            whss_recall /= nq as f32;
            let whss_qps = nq as f64 / (whss_ms / 1000.0);

            // SRP search
            let ts = Instant::now();
            let mut srp_recall = 0.0f32;
            for (i, q) in queries.iter().enumerate() {
                let pred = srp_search(&srp, &corpus, &ssk, q, k, cand);
                srp_recall += recall_at_k(&truths[i], &pred);
            }
            let srp_ms = ts.elapsed().as_secs_f64() * 1000.0;
            srp_recall /= nq as f32;
            let srp_qps = nq as f64 / (srp_ms / 1000.0);

            println!(
                "WHSS m={:>3} cand={:>5}: {:>8.2} ms | {:>8.1} QPS | recall={:.4} | build={:>7.1} ms | sketch_bytes/vec={}",
                m_bits, cand, whss_ms, whss_qps, whss_recall, whss_build_ms, m_bits / 8
            );
            println!(
                "SRP  m={:>3} cand={:>5}: {:>8.2} ms | {:>8.1} QPS | recall={:.4} | build={:>7.1} ms | model_bytes={}",
                m_bits, cand, srp_ms, srp_qps, srp_recall, srp_build_ms, m_bits * d * 4
            );
        }
    }

    // Memory summary
    println!(
        "\n# Memory\nraw float32 corpus : {:>10} bytes",
        n * d * 4
    );
    for &m in &[128usize, 256, 512] {
        println!("WHSS sketches m={:>3}: {:>10} bytes ({:>6.2}x compression)", m, n * (m / 8), (n * d * 4) as f64 / (n * (m / 8)) as f64);
    }
}
