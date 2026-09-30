//! Benchmark: encode / decode / IVF-scan-sim over Raw, VByte, Elias-Fano.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruvector_ef_ivf_posting::{EliasFano, PostingCodec, RawU32, VByteDelta};
use std::time::Instant;

fn gen_ivf_lists(n_lists: usize, avg_len: usize, universe: u32, seed: u64) -> Vec<Vec<u32>> {
    let mut r = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n_lists);
    for _ in 0..n_lists {
        let len = ((avg_len as f64) * (0.5 + r.gen::<f64>())) as usize;
        let mut ids: Vec<u32> = (0..len).map(|_| r.gen_range(0..universe)).collect();
        ids.sort();
        ids.dedup();
        out.push(ids);
    }
    out
}

struct Stat {
    codec: &'static str,
    total_bytes: usize,
    encode_ns: u128,
    decode_ns: u128,
    scan_ns: u128,
    raw_bytes: usize,
    n_ids: usize,
}

fn run<C: PostingCodec + ?Sized>(codec: &C, lists: &[Vec<u32>], scan_iters: usize) -> Stat {
    // Encode
    let t = Instant::now();
    let encoded: Vec<Vec<u8>> = lists.iter().map(|l| codec.encode(l)).collect();
    let encode_ns = t.elapsed().as_nanos();
    let total_bytes: usize = encoded.iter().map(|e| e.len()).sum();
    let raw_bytes: usize = lists.iter().map(|l| l.len() * 4).sum();
    let n_ids: usize = lists.iter().map(|l| l.len()).sum();

    // Full decode
    let t = Instant::now();
    let mut buf = Vec::with_capacity(4096);
    let mut checksum: u64 = 0;
    for e in &encoded {
        codec.decode(e, &mut buf);
        for &x in &buf { checksum = checksum.wrapping_add(x as u64); }
    }
    let decode_ns = t.elapsed().as_nanos();
    std::hint::black_box(checksum);

    // Scan-simulation: decode a subset of lists many times (query pattern: nprobe lists).
    let t = Instant::now();
    let mut r = ChaCha8Rng::seed_from_u64(0xC0FFEE);
    let mut acc: u64 = 0;
    for _ in 0..scan_iters {
        let i = r.gen_range(0..encoded.len());
        codec.decode(&encoded[i], &mut buf);
        for &x in &buf { acc = acc.wrapping_add(x as u64); }
    }
    let scan_ns = t.elapsed().as_nanos();
    std::hint::black_box(acc);

    Stat {
        codec: codec.name(),
        total_bytes,
        encode_ns,
        decode_ns,
        scan_ns,
        raw_bytes,
        n_ids,
    }
}

fn fmt_ns(ns: u128, n: usize) -> String {
    if n == 0 { return "n/a".into(); }
    format!("{:.1} ns/id", ns as f64 / n as f64)
}

fn main() {
    let scenarios = [
        ("dense/uniform", 256usize, 4_000usize, 1_000_000u32),
        ("sparse/wide",   1024usize, 500usize,  10_000_000u32),
        ("very-dense",    64usize,   16_000usize, 1_500_000u32),
    ];
    let scan_iters = 20_000usize;

    for (name, n_lists, avg_len, universe) in scenarios {
        let lists = gen_ivf_lists(n_lists, avg_len, universe, 7);
        let total_ids: usize = lists.iter().map(|l| l.len()).sum();
        println!("\n=== {} : {} lists, ~{} ids/list, universe {} (total {} ids) ===",
                 name, n_lists, avg_len, universe, total_ids);

        let codecs: Vec<Box<dyn PostingCodec>> =
            vec![Box::new(RawU32), Box::new(VByteDelta), Box::new(EliasFano)];
        let mut stats = Vec::new();
        for c in &codecs {
            stats.push(run(c.as_ref(), &lists, scan_iters));
        }

        println!(
            "{:<14} {:>14} {:>10} {:>14} {:>16} {:>16}",
            "codec", "bytes", "bits/id", "compress×", "decode", "scan"
        );
        for s in &stats {
            let bpi = (s.total_bytes as f64 * 8.0) / s.n_ids as f64;
            let ratio = s.raw_bytes as f64 / s.total_bytes as f64;
            println!(
                "{:<14} {:>14} {:>10.2} {:>13.2}× {:>16} {:>16}",
                s.codec, s.total_bytes, bpi, ratio,
                fmt_ns(s.decode_ns, s.n_ids),
                fmt_ns(s.scan_ns, scan_iters * (s.n_ids / n_lists.max(1)).max(1)),
            );
            let _ = s.encode_ns;
        }
    }

    // Sanity: correctness across all codecs on the same data.
    let lists = gen_ivf_lists(32, 800, 100_000, 123);
    let raw_all: Vec<Vec<u32>> = lists.clone();
    for codec_name in ["raw", "vb", "ef"] {
        let mut roundtrip: Vec<Vec<u32>> = Vec::new();
        let mut buf = Vec::new();
        for l in &lists {
            let bytes = match codec_name {
                "raw" => RawU32.encode(l),
                "vb"  => VByteDelta.encode(l),
                "ef"  => EliasFano.encode(l),
                _ => unreachable!(),
            };
            match codec_name {
                "raw" => RawU32.decode(&bytes, &mut buf),
                "vb"  => VByteDelta.decode(&bytes, &mut buf),
                "ef"  => EliasFano.decode(&bytes, &mut buf),
                _ => unreachable!(),
            }
            roundtrip.push(buf.clone());
        }
        assert_eq!(roundtrip, raw_all, "codec {} roundtrip mismatch", codec_name);
    }
    println!("\nCorrectness: all codecs roundtrip identical on 32×800 lists.");
}
