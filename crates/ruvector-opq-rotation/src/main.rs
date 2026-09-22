//! opq-benchmark: measures recall@10 and per-query latency of PlainPQ vs
//! RandomRotPQ vs OpqHouseholder on synthetic skewed-variance data.

use ruvector_opq_rotation::{
    util::{brute_topk, gauss_skewed, recall_at_k},
    OpqHouseholder, PlainPq, Quantizer, RandomRotPq,
};
use std::time::Instant;

fn eval<Q: Quantizer>(
    name: &str,
    q: &Q,
    data: &[Vec<f32>],
    queries: &[Vec<f32>],
    gt: &[Vec<usize>],
    k: usize,
) {
    // encode all vectors
    let t_enc = Instant::now();
    let codes: Vec<Vec<u8>> = data.iter().map(|v| q.encode(v)).collect();
    let enc_ms = t_enc.elapsed().as_secs_f64() * 1000.0;

    // search
    let mut recall_sum = 0f32;
    let t_search = Instant::now();
    for (qi, qv) in queries.iter().enumerate() {
        let tables = q.adc_tables(qv);
        let mut scored: Vec<(f32, usize)> = codes
            .iter()
            .enumerate()
            .map(|(i, c)| (q.adc_distance(c, &tables), i))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let approx: Vec<usize> = scored.into_iter().take(k).map(|(_, i)| i).collect();
        recall_sum += recall_at_k(&gt[qi], &approx, k);
    }
    let search_ms = t_search.elapsed().as_secs_f64() * 1000.0;
    let per_q = search_ms / queries.len() as f64;

    let recall = recall_sum / queries.len() as f32;
    println!(
        "{:<20}  recall@{}={:.4}  encode={:>6.1}ms  search_total={:>6.1}ms  per_query={:>5.2}ms",
        name, k, recall, enc_ms, search_ms, per_q
    );
}

fn main() {
    let d = 64;
    let n = 4000;
    let n_q = 100;
    let k = 10;
    let m = 8;
    let ks = 256; // standard byte-code PQ: 8 bytes/vector, 8× compression from f32

    println!(
        "== ruvector-opq-rotation benchmark ==\nd={} n={} n_q={} k={} m={} ks={} (bytes/code={})",
        d, n, n_q, k, m, ks, m
    );

    let data = gauss_skewed(n, d, 42, 4.0);
    let queries = gauss_skewed(n_q, d, 1337, 4.0);

    // ground truth
    let t_gt = Instant::now();
    let gt: Vec<Vec<usize>> = queries.iter().map(|q| brute_topk(&data, q, k)).collect();
    println!("ground truth built in {:.1}ms", t_gt.elapsed().as_secs_f64() * 1000.0);

    // baseline PQ
    let t = Instant::now();
    let mut plain = PlainPq::new(d, m, ks, 20, 7);
    plain.train(&data);
    println!("PlainPq trained in {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    eval("PlainPq", &plain, &data, &queries, &gt, k);

    // random rotation
    let t = Instant::now();
    let mut rrp = RandomRotPq::build(d, m, ks, 20, 8);
    rrp.train(&data);
    println!("RandomRotPq trained in {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    eval("RandomRotPq", &rrp, &data, &queries, &gt, k);

    // OPQ
    let t = Instant::now();
    let opq = OpqHouseholder::build(d, m, ks, 20, 3, &data, 9);
    println!("OpqHouseholder trained in {:.1}ms", t.elapsed().as_secs_f64() * 1000.0);
    eval("OpqHouseholder", &opq, &data, &queries, &gt, k);
}
