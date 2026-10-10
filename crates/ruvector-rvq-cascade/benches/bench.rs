//! Measured comparison: PQ-only vs PQ+full-verify vs RVQ-cascade.
//!
//! The RVQ cascade's thesis: at a target recall, it needs FEWER
//! full-precision vector fetches than a classic PQ -> full-verify
//! rerank. Full-precision fetches dominate cost when vectors live on
//! disk / NVMe or when d is large (SIMD doesn't help compressed codes).
//! We measure that fetch count directly.
//!
//! Scalar portable Rust; numbers reproduce bit-exactly on any box.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use ruvector_rvq_cascade::*;
use std::time::Instant;

fn gen(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let n_clusters = 100;
    let centers: Vec<f32> = (0..n_clusters * dim)
        .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
        .collect();
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = rng.gen_range(0..n_clusters);
        for j in 0..dim {
            let noise: f32 = rng.gen::<f32>() * 0.3 - 0.15;
            out.push(centers[c * dim + j] + noise);
        }
    }
    out
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Classic PQ -> full-verify rerank with target shortlist N.
fn pq_rerank(
    pq: &Pq8,
    data: &[f32],
    dim: usize,
    n: usize,
    q: &[f32],
    k: usize,
    shortlist: usize,
) -> Vec<u32> {
    let lut = pq.l2_luts(q);
    let mut cand: Vec<(u32, f32)> =
        (0..n as u32).map(|i| (i, pq.adc(i as usize, &lut))).collect();
    let keep = shortlist.min(cand.len());
    cand.select_nth_unstable_by(keep - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    cand.truncate(keep);
    for p in cand.iter_mut() {
        let v = &data[p.0 as usize * dim..(p.0 as usize + 1) * dim];
        p.1 = l2sq(q, v);
    }
    cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    cand.truncate(k);
    cand.into_iter().map(|x| x.0).collect()
}

fn main() {
    let n = 20_000usize;
    let dim = 128usize;
    let nq = 200usize;
    let k = 10usize;
    let data = gen(n, dim, 1);
    let queries = gen(nq, dim, 2);

    println!("=== RVQ cascade bench (scalar, N={n}, d={dim}, nq={nq}, k={k}) ===");

    // Train
    let t = Instant::now();
    let pq = Pq8::train(&data, dim, 8, 15, 42);
    let pq_train = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let prvq = Prvq::train(&data, dim, 8, 4, 12, 42);
    let prvq_train = t.elapsed().as_secs_f64();

    println!(
        "Train: PQ-8: {:.3}s  PRVQ (m=8, K=4): {:.3}s",
        pq_train, prvq_train
    );
    println!(
        "Memory: PQ = {} KB  PRVQ = {} KB  float32 raw = {} KB",
        pq.memory_bytes() / 1024,
        prvq.memory_bytes() / 1024,
        n * dim * 4 / 1024
    );

    // Ground truth
    let gts: Vec<Vec<u32>> = (0..nq)
        .map(|qi| {
            let q = &queries[qi * dim..(qi + 1) * dim];
            brute_force_topk(&data, dim, q, k)
                .into_iter()
                .map(|x| x.0)
                .collect()
        })
        .collect();

    // Variant A: Full float32 brute
    let t = Instant::now();
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let _ = brute_force_topk(&data, dim, q, k);
    }
    let full_ms = t.elapsed().as_secs_f64() * 1000.0 / nq as f64;

    // Variant B: PQ-ADC only
    let t = Instant::now();
    let mut pq_only_recalls = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let lut = pq.l2_luts(q);
        let mut cand: Vec<(u32, f32)> =
            (0..n as u32).map(|i| (i, pq.adc(i as usize, &lut))).collect();
        cand.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        cand.truncate(k);
        let r: Vec<u32> = cand.into_iter().map(|x| x.0).collect();
        pq_only_recalls.push(recall_at_k(&r, &gts[qi]) as f64);
    }
    let pq_ms = t.elapsed().as_secs_f64() * 1000.0 / nq as f64;

    // Variant C: PQ -> full-verify at several shortlist sizes.
    // Sweep to find smallest N reaching recall >= 0.95.
    let mut classic_results: Vec<(usize, f64, f64)> = Vec::new();
    for &sl in &[20usize, 50, 100, 200, 400, 800, 1600] {
        let t = Instant::now();
        let mut rec = Vec::with_capacity(nq);
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let r = pq_rerank(&pq, &data, dim, n, q, k, sl);
            rec.push(recall_at_k(&r, &gts[qi]) as f64);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / nq as f64;
        classic_results.push((sl, mean(&rec), ms));
    }

    // Variant D: PRVQ pipeline — PQ+RVQ-on-residual as refined
    // estimator; prune a large PQ shortlist with PRVQ, full-verify the
    // winners. Explore different (pq_shortlist, final_verify) counts.
    let mut cascade_results: Vec<((usize, usize), f64, f64)> = Vec::new();
    for &(sl_pq, sl_final) in &[
        (2000usize, 20usize),
        (2000, 50),
        (2000, 100),
        (2000, 200),
        (2000, 400),
    ] {
        let t = Instant::now();
        let mut rec = Vec::with_capacity(nq);
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            // Stage 0: PQ ADC filter -> top sl_pq
            let pq_lut = prvq.pq.l2_luts(q);
            let mut cand: Vec<(u32, f32)> = (0..n as u32)
                .map(|i| (i, prvq.pq.adc(i as usize, &pq_lut)))
                .collect();
            let keep = sl_pq.min(cand.len());
            cand.select_nth_unstable_by(keep - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(keep);
            // Stage 1: PRVQ refine -> top sl_final
            let dot_luts = prvq.rvq.build_dot_luts(q);
            for p in cand.iter_mut() {
                p.1 = prvq.estimate_l2sq(p.0 as usize, &pq_lut, &dot_luts);
            }
            let keep = sl_final.min(cand.len());
            cand.select_nth_unstable_by(keep - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(keep);
            // Stage 2: full-verify
            for p in cand.iter_mut() {
                let v = &data[p.0 as usize * dim..(p.0 as usize + 1) * dim];
                p.1 = l2sq(q, v);
            }
            cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(k);
            let r: Vec<u32> = cand.into_iter().map(|x| x.0).collect();
            rec.push(recall_at_k(&r, &gts[qi]) as f64);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / nq as f64;
        cascade_results.push(((sl_pq, sl_final), mean(&rec), ms));
    }

    println!("\n{:<42} {:>10} {:>12}", "Variant", "ms/query", "recall@10");
    println!("{}", "-".repeat(66));
    println!("{:<42} {:>10.3} {:>12.3}", "A. Full float32 brute", full_ms, 1.0);
    println!(
        "{:<42} {:>10.3} {:>12.3}",
        "B. PQ-ADC only (no rerank)",
        pq_ms,
        mean(&pq_only_recalls)
    );
    for (sl, rec, ms) in &classic_results {
        println!(
            "{:<42} {:>10.3} {:>12.3}",
            format!("C. PQ -> full-verify top-{} (fetches={})", sl, sl),
            ms,
            rec
        );
    }
    for (sh, rec, ms) in &cascade_results {
        println!(
            "{:<42} {:>10.3} {:>12.3}",
            format!(
                "D. PRVQ cascade pq={} verify={}",
                sh.0, sh.1
            ),
            ms,
            rec
        );
    }

    // Headline: at matched recall, which has fewer float32 fetches?
    let target = 0.95;
    let class = classic_results
        .iter()
        .find(|(_, r, _)| *r >= target)
        .copied();
    let casc: Option<((usize, usize), f64, f64)> = cascade_results
        .iter()
        .find(|(_, r, _)| *r >= target)
        .copied();
    println!("\nAt recall >= {:.2}:", target);
    if let Some((sl, rec, ms)) = class {
        println!(
            "  Classic PQ-rerank: top-{} shortlist -> {} fetches, {:.3} ms, recall {:.3}",
            sl, sl, ms, rec
        );
    } else {
        println!("  Classic PQ-rerank: did not reach target recall in sweep");
    }
    if let Some((sh, rec, ms)) = casc {
        println!(
            "  PRVQ cascade: pq={} verify={} -> {:.3} ms, recall {:.3}",
            sh.0, sh.1, ms, rec
        );
    } else {
        println!("  PRVQ cascade: did not reach target recall in sweep");
    }
    if let (Some((sl_c, _, _)), Some((sh_r, _, _))) = (class, casc) {
        let fetch_ratio = sl_c as f64 / sh_r.1 as f64;
        println!(
            "  Fetch reduction (PRVQ vs classic at same recall): {:.1}x fewer full-precision fetches",
            fetch_ratio
        );
    }
}
