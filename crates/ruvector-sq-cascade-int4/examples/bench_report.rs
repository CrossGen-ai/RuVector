//! Real end-to-end benchmark: builds a 50k x 128 dataset, runs 1000 queries,
//! reports recall@10, MSE/dim, bytes/vec, encode/decode ns for fp32 / int8 / cascade.
//!
//! Run: cargo run --release -p ruvector-sq-cascade-int4 --example bench_report

use ruvector_sq_cascade_int4::{
    dataset::{gaussian_dataset, uniform_queries},
    eval::evaluate,
    quant::{measure, CascadeInt8Int4, Fp32Baseline, Int8Sq, Quantizer},
};

fn main() {
    let n = 50_000;
    let d = 128;
    let k = 10;
    let n_q = 1_000;

    println!("## Dataset");
    println!("N={}  D={}  k={}  queries={}", n, d, k, n_q);
    let base = gaussian_dataset(n, d, 0xC0FFEE);
    let queries = uniform_queries(n_q, d, 0xBADF00D);

    // --- fp32 baseline (encode/decode is bit-copy; recall = 1.0 by definition if dataset unchanged) ---
    let fp = Fp32Baseline::new(d);
    let fp_stats = measure(&fp, &base);

    // --- int8 ---
    let mut i8 = Int8Sq::new(d);
    i8.train(&base);
    let i8_stats = measure(&i8, &base);
    let i8_eval = evaluate(&i8, &base, &queries, k);

    // --- cascade ---
    let mut casc = CascadeInt8Int4::new(d);
    casc.train(&base);
    let casc_stats = measure(&casc, &base);
    let casc_eval = evaluate(&casc, &base, &queries, k);

    println!("\n## Results (release)");
    println!("| variant             | bytes/vec | enc ns/vec | dec ns/vec | recall@10 | mse/dim      |");
    println!("|---------------------|-----------|------------|------------|-----------|--------------|");
    println!("| fp32                | {:9} | {:10.1} | {:10.1} | 1.000     | 0            |",
             fp_stats.bytes_per_vec, fp_stats.encode_ns_per_vec, fp_stats.decode_ns_per_vec);
    println!("| int8_sq             | {:9} | {:10.1} | {:10.1} | {:.3}     | {:.4e}   |",
             i8_stats.bytes_per_vec, i8_stats.encode_ns_per_vec, i8_stats.decode_ns_per_vec,
             i8_eval.recall_at_k, i8_eval.avg_mse_per_dim);
    println!("| cascade_int8_int4   | {:9} | {:10.1} | {:10.1} | {:.3}     | {:.4e}   |",
             casc_stats.bytes_per_vec, casc_stats.encode_ns_per_vec, casc_stats.decode_ns_per_vec,
             casc_eval.recall_at_k, casc_eval.avg_mse_per_dim);

    // Totals
    let gb_fp = (n * fp_stats.bytes_per_vec) as f64 / 1_073_741_824.0;
    let gb_i8 = (n * i8_stats.bytes_per_vec) as f64 / 1_073_741_824.0;
    let gb_c  = (n * casc_stats.bytes_per_vec) as f64 / 1_073_741_824.0;
    println!("\n## Memory totals (N={})", n);
    println!("fp32:    {:>8.4} GiB", gb_fp);
    println!("int8:    {:>8.4} GiB", gb_i8);
    println!("cascade: {:>8.4} GiB", gb_c);

    let mse_ratio = i8_eval.avg_mse_per_dim / casc_eval.avg_mse_per_dim;
    let bytes_ratio = (casc_stats.bytes_per_vec as f64) / (i8_stats.bytes_per_vec as f64);
    println!("\n## Ratios");
    println!("mse(int8)/mse(cascade) = {:.2}x   (higher → cascade has less error)", mse_ratio);
    println!("bytes(cascade)/bytes(int8) = {:.2}x", bytes_ratio);
    println!("recall@10 gain: {:+.4}", casc_eval.recall_at_k - i8_eval.recall_at_k);
}
