//! Real measurements for the three posting-list codecs.
//!
//! Simulates an IVF index over N docs partitioned into C posting lists
//! (clusters). For each codec we report:
//!   - total bytes and bytes/doc_id
//!   - decode-only throughput (M ids / sec, averaged over repeated passes)
//!   - end-to-end scan (decode + L2 top-K) latency per cluster
//!   - top-K result identity vs the raw baseline (should match exactly).

use ruvector_ivf_pfor_postings::{pfor::PforCodec, raw::RawCodec, scan, varint::VarintCodec, PostingCodec};
use std::time::Instant;

fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn build_postings(n_docs: u32, n_clusters: u32, seed: u64) -> Vec<Vec<u32>> {
    // Assign each doc id to a random cluster, then sort within each list.
    let mut lists = vec![Vec::<u32>::new(); n_clusters as usize];
    let mut st = seed | 1;
    for id in 0..n_docs {
        let c = (xorshift(&mut st) % n_clusters as u64) as usize;
        lists[c].push(id);
    }
    // Already sorted since we pushed ids in increasing order.
    lists
}

fn build_vectors(n_docs: u32, dim: usize, seed: u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(n_docs as usize * dim);
    let mut st = seed | 1;
    for _ in 0..n_docs as usize * dim {
        let r = xorshift(&mut st);
        let f = ((r & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
        v.push(f);
    }
    v
}

fn report_codec<C: PostingCodec>(
    label: &str,
    lists: &[Vec<u32>],
) -> (usize, Vec<Vec<u8>>) {
    let t0 = Instant::now();
    let encoded: Vec<Vec<u8>> = lists.iter().map(|l| C::encode(l)).collect();
    let enc_elapsed = t0.elapsed();
    let total_bytes: usize = encoded.iter().map(|b| b.len()).sum();
    let total_ids: usize = lists.iter().map(|l| l.len()).sum();
    let bits_per_id = (total_bytes as f64 * 8.0) / total_ids.max(1) as f64;
    println!(
        "  encode: {:>14} | {:>10} bytes | {:>6.2} b/id | encode-all: {:>7.2} ms",
        label,
        total_bytes,
        bits_per_id,
        enc_elapsed.as_secs_f64() * 1000.0
    );
    (total_bytes, encoded)
}

fn measure_decode_throughput<C: PostingCodec>(
    label: &str,
    encoded: &[Vec<u8>],
    total_ids: usize,
    passes: usize,
) {
    let mut sink: u64 = 0;
    let t0 = Instant::now();
    for _ in 0..passes {
        for buf in encoded {
            let ids = C::decode(buf);
            for id in ids {
                sink = sink.wrapping_add(id as u64);
            }
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    let m_ids = (total_ids as f64 * passes as f64) / 1e6;
    let m_ids_per_sec = m_ids / dt;
    println!(
        "  decode: {:>14} | {:>7.1} M ids/s | sink={}",
        label, m_ids_per_sec, sink
    );
}

fn measure_scan<C: PostingCodec>(
    label: &str,
    encoded: &[Vec<u8>],
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    k: usize,
) -> (f64, Vec<Vec<scan::Hit>>) {
    let t0 = Instant::now();
    let mut results = Vec::with_capacity(encoded.len());
    for buf in encoded {
        let ids = C::decode(buf);
        let hits = scan::scan(&ids, vectors, dim, query, k);
        results.push(hits);
    }
    let dt = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  scan:   {:>14} | all clusters: {:>7.2} ms", label, dt);
    (dt, results)
}

fn main() {
    println!("=== ruvector-ivf-pfor-postings — real measurements ===\n");

    let configs = [
        ("tiny",   100_000u32,   128u32),
        ("medium", 1_000_000,    512),
        ("large",  5_000_000,   2048),
    ];
    let dim = 64usize;

    for (name, n_docs, n_clusters) in configs {
        println!(
            "--- scenario: {} ({} docs, {} clusters, avg {:.1} ids/list) ---",
            name,
            n_docs,
            n_clusters,
            n_docs as f64 / n_clusters as f64
        );
        let lists = build_postings(n_docs, n_clusters, 0xBEEF_CAFE);
        let total_ids: usize = lists.iter().map(|l| l.len()).sum();

        let (raw_bytes, raw_enc) = report_codec::<RawCodec>("raw-u32", &lists);
        let (vi_bytes, vi_enc) = report_codec::<VarintCodec>("delta+varint", &lists);
        let (pf_bytes, pf_enc) = report_codec::<PforCodec>("delta+pfor-128", &lists);
        println!(
            "  compression: varint {:.2}x, pfor {:.2}x",
            raw_bytes as f64 / vi_bytes.max(1) as f64,
            raw_bytes as f64 / pf_bytes.max(1) as f64,
        );

        // Decode throughput — pick pass count that keeps wall-clock reasonable.
        let passes = if n_docs <= 100_000 {
            50
        } else if n_docs <= 1_000_000 {
            5
        } else {
            2
        };
        measure_decode_throughput::<RawCodec>("raw-u32", &raw_enc, total_ids, passes);
        measure_decode_throughput::<VarintCodec>("delta+varint", &vi_enc, total_ids, passes);
        measure_decode_throughput::<PforCodec>("delta+pfor-128", &pf_enc, total_ids, passes);

        // End-to-end scan over all clusters (acts as a worst-case "scan the whole index").
        let vectors = build_vectors(n_docs, dim, 0x1234_5678);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32).sin() * 0.5).collect();
        let k = 10usize;
        let (_, raw_hits) = measure_scan::<RawCodec>("raw-u32", &raw_enc, &vectors, dim, &query, k);
        let (_, vi_hits) = measure_scan::<VarintCodec>("delta+varint", &vi_enc, &vectors, dim, &query, k);
        let (_, pf_hits) = measure_scan::<PforCodec>("delta+pfor-128", &pf_enc, &vectors, dim, &query, k);

        // Correctness: top-K identity across codecs (must match exactly).
        for (i, (a, b)) in raw_hits.iter().zip(vi_hits.iter()).enumerate() {
            assert_eq!(a, b, "varint top-K differs from raw at cluster {}", i);
        }
        for (i, (a, b)) in raw_hits.iter().zip(pf_hits.iter()).enumerate() {
            assert_eq!(a, b, "pfor top-K differs from raw at cluster {}", i);
        }
        println!("  correctness: top-{k} identical across all three codecs for all clusters\n");
    }
    println!("done.");
}
