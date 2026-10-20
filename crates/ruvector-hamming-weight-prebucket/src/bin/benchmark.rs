//! Real benchmark: synthesize a dataset, encode with SimHash, measure
//! wall-clock knn QPS and candidates-visited for the three scanners.
//! All numbers come from this binary. No mocks.

use ruvector_hamming_weight_prebucket::*;
use std::time::Instant;

fn gen_vectors(n: usize, dim: usize, cluster_frac: f32, seed: u64) -> Vec<Vec<f32>> {
    // Build a mix of random Gaussian and clustered vectors so SimHash bits
    // show non-uniform popcount — the regime where weight-bucketing helps.
    let mut rng = SplitMix64::new(seed);
    let n_clustered = (n as f32 * cluster_frac) as usize;
    let n_rand = n - n_clustered;
    let mut out = Vec::with_capacity(n);
    let n_centers = 16usize;
    let centers: Vec<Vec<f32>> = (0..n_centers).map(|_| {
        (0..dim).map(|_| rng.gauss() * 2.0).collect()
    }).collect();
    for _ in 0..n_clustered {
        let c = ((rng.next_u64() as usize) % n_centers) as usize;
        let v: Vec<f32> = centers[c].iter().map(|m| m + 0.5 * rng.gauss()).collect();
        out.push(v);
    }
    for _ in 0..n_rand {
        let v: Vec<f32> = (0..dim).map(|_| rng.gauss()).collect();
        out.push(v);
    }
    out
}

fn mean(v: &[f64]) -> f64 { v.iter().sum::<f64>() / v.len() as f64 }

fn run_variant<const W: usize, S: BinaryScanner<W>>(
    label: &str,
    scanner: &S,
    queries: &[Code<W>],
    k: usize,
) -> (f64, f64, f64) {
    // Warm-up
    for q in queries.iter().take(32) { let _ = scanner.knn(q, k); }
    scanner.reset_counters();
    let t0 = Instant::now();
    let mut vis_total: f64 = 0.0;
    for q in queries {
        scanner.reset_counters();
        let _ = scanner.knn(q, k);
        vis_total += scanner.visited() as f64;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let qps = queries.len() as f64 / elapsed;
    let avg_visits = vis_total / queries.len() as f64;
    println!(
        "  {:<28} qps={:>9.1}  avg_visits={:>9.1}  total_s={:.3}",
        label, qps, avg_visits, elapsed
    );
    (qps, avg_visits, elapsed)
}

/// Distance-set recall: fraction of returned items whose Hamming distance
/// appears in the ground-truth distance multiset. This is the exactness
/// measure that is robust to Hamming ties (very common on binary codes).
fn recall(ground: &[(u32, u32)], got: &[(u32, u32)]) -> f64 {
    let mut gd: Vec<u32> = ground.iter().map(|(_, d)| *d).collect();
    gd.sort();
    let mut ad: Vec<u32> = got.iter().map(|(_, d)| *d).collect();
    ad.sort();
    if gd == ad { 1.0 } else { 0.0 }
}

fn run_regime(label: &str, codes: Vec<Code<2>>, qcodes: Vec<Code<2>>, k: usize) {
    println!("\n==== regime: {} ====", label);
    let bits = 128usize;
    let n = codes.len();

    let mut hist = vec![0u64; bits + 1];
    for c in codes.iter() { hist[popcount(c) as usize] += 1; }
    let hmin = hist.iter().position(|&x| x > 0).unwrap();
    let hmax = hist.iter().rposition(|&x| x > 0).unwrap();
    let hmean = hist.iter().enumerate().map(|(w, &c)| (w as f64) * (c as f64)).sum::<f64>() / n as f64;
    // Estimate stddev.
    let var = hist.iter().enumerate()
        .map(|(w, &c)| { let dw = w as f64 - hmean; dw * dw * (c as f64) })
        .sum::<f64>() / n as f64;
    println!("  popcount[min={}, mean={:.2}, stddev={:.2}, max={}]", hmin, hmean, var.sqrt(), hmax);

    let full = FullScan::<2>::new(codes.clone());
    let wb = WeightedScan::<2>::build(codes.clone());
    let wbs = WeightedBitSlice::<2>::build(codes.clone());

    let n_check = 128.min(qcodes.len());
    let mut r_wb = Vec::with_capacity(n_check);
    let mut r_wbs = Vec::with_capacity(n_check);
    for q in qcodes.iter().take(n_check) {
        let g = full.knn(q, k);
        let a = wb.knn(q, k);
        let b = wbs.knn(q, k);
        r_wb.push(recall(&g, &a));
        r_wbs.push(recall(&g, &b));
    }
    let avg_r_wb = mean(&r_wb);
    let avg_r_wbs = mean(&r_wbs);
    println!("  exactness: weight-bucket={:.4}  weight+bitslice={:.4}", avg_r_wb, avg_r_wbs);
    assert!((avg_r_wb - 1.0).abs() < 1e-9, "weight-bucket must be exact in {}", label);
    assert!((avg_r_wbs - 1.0).abs() < 1e-9, "weight+bitslice must be exact in {}", label);

    let (full_qps, full_vis, _) = run_variant("full-scan", &full, &qcodes, k);
    let (wb_qps, wb_vis, _) = run_variant("weight-bucket", &wb, &qcodes, k);
    let (wbs_qps, wbs_vis, _) = run_variant("weight+bitslice", &wbs, &qcodes, k);

    let bucket_mem = bucket_bytes::<2>(&wb.buckets);
    let codes_mem = n * std::mem::size_of::<Code<2>>();
    println!(
        "  memory: codes={:.2} MB, buckets={:.2} MB, overhead={:.1}%",
        codes_mem as f64 / 1_048_576.0, bucket_mem as f64 / 1_048_576.0,
        100.0 * bucket_mem as f64 / codes_mem as f64,
    );
    println!(
        "  speedup wb/full = {:.2}x  wbs/full = {:.2}x | visit-prune wb/full = {:.2}x  wbs/full = {:.2}x",
        wb_qps / full_qps, wbs_qps / full_qps,
        full_vis / wb_vis.max(1.0), full_vis / wbs_vis.max(1.0)
    );
}

/// Encode a vector with "biased" SimHash: each projection has a random offset
/// b_i applied before the sign — bits take value sign(<r,x> - b_i). Non-zero
/// offsets make the per-bit activation probability drift away from 0.5 and
/// produce codes with a wide popcount distribution.
fn simhash_biased<const W: usize>(x: &[f32], signs: &SignTable, biases: &[f32]) -> Code<2> {
    let bits = W * 64;
    let mut code = [0u64; 2];
    for b in 0..bits {
        let row = &signs.rows[b];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            let s = ((row[i / 64] >> (i % 64)) & 1) as i32;
            let sign = (s << 1) - 1;
            acc += (sign as f32) * xi;
        }
        if acc >= biases[b] {
            code[b / 64] |= 1u64 << (b % 64);
        }
    }
    code
}

fn main() {
    let dim = 128usize;
    let bits = 128usize;
    let n = 100_000usize;
    let n_q = 1_000usize;
    let k = 10usize;

    println!("ruvector-hamming-weight-prebucket :: real benchmark");
    println!(
        "config: n={}, dim={}, bits={}, k={}, queries={}",
        n, dim, bits, k, n_q
    );

    let t0 = Instant::now();
    let vecs = gen_vectors(n, dim, 0.7, 0xA5A5);
    let q_vecs = gen_vectors(n_q, dim, 0.7, 0x1234);
    println!("data gen: {:.3}s", t0.elapsed().as_secs_f64());

    let signs = SignTable::new(dim, bits, 0xC0DE);

    // Regime A: unbiased SimHash. Honest baseline — popcount concentrates near
    // D/2 with stddev ~ sqrt(D)/2, so the pre-bucket layer rarely prunes
    // and the full scan can win on sheer loop locality.
    let t0 = Instant::now();
    let codes_a: Vec<Code<2>> = vecs.iter().map(|v| simhash_encode::<2>(v, &signs)).collect();
    let qcodes_a: Vec<Code<2>> = q_vecs.iter().map(|v| simhash_encode::<2>(v, &signs)).collect();
    println!("encode A: {:.3}s", t0.elapsed().as_secs_f64());
    run_regime("unbiased SimHash (D/2 symmetric)", codes_a, qcodes_a, k);

    // Regime B: class-structured codes. Three "modes" at popcounts 25 / 64 /
    // 100. Queries draw from one of the modes. This models learned sparse
    // codes (e.g. one-bit per bit-group) or heterogeneous embeddings. Here
    // tau is small vs the inter-mode gap and triangle pruning cuts hard.
    let mut rng = SplitMix64::new(0xFACE);
    let mk = |target: usize, rng: &mut SplitMix64| -> Code<2> {
        let mut c = [0u64; 2];
        let mut set = 0usize;
        while set < target {
            let b = (rng.next_u64() % 128) as usize;
            let mask = 1u64 << (b % 64);
            if c[b / 64] & mask == 0 { c[b / 64] |= mask; set += 1; }
        }
        c
    };
    // Build 100k codes with cluster-correlated noise: pick a mode center
    // (random code at target popcount), then jitter 4 bits.
    let modes = [25usize, 64usize, 100usize];
    let centers: Vec<Code<2>> = modes.iter().map(|&t| mk(t, &mut rng)).collect();
    let codes_b: Vec<Code<2>> = (0..n).map(|_| {
        let m = (rng.next_u64() % modes.len() as u64) as usize;
        let mut c = centers[m];
        for _ in 0..4 {
            let b = (rng.next_u64() % 128) as usize;
            c[b / 64] ^= 1u64 << (b % 64);
        }
        c
    }).collect();
    let qcodes_b: Vec<Code<2>> = (0..n_q).map(|_| {
        let m = (rng.next_u64() % modes.len() as u64) as usize;
        let mut c = centers[m];
        for _ in 0..4 {
            let b = (rng.next_u64() % 128) as usize;
            c[b / 64] ^= 1u64 << (b % 64);
        }
        c
    }).collect();
    run_regime("class-structured modes {25,64,100}", codes_b, qcodes_b, k);
}
