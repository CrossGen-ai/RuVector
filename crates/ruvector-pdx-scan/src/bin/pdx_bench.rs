//! Real benchmark for the three backends. Prints wall-clock ms, throughput,
//! recall vs. the horizontal brute-force baseline, PDX prune stats, and IVF
//! numbers. No mocks: all timings come from the real code paths.

use rand::prelude::*;
use rand_distr::StandardNormal;
use ruvector_pdx_scan::{
    brute_horizontal, brute_pdx, brute_pdx_pruned, brute_pdx_block_pruned,
    ivf::{IvfHorizontal, IvfPdxPruned},
    HorizontalStore, PdxStore,
};
use ruvector_pdx_scan::pruned::DimensionOrder;
use std::time::Instant;

fn make(n: usize, d: usize, seed: u64) -> (HorizontalStore, Vec<Vec<f32>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..d).map(|_| rng.sample::<f32, _>(StandardNormal)).collect())
        .collect();
    (HorizontalStore::from_iter(d, vecs), queries)
}

fn ids_of(v: &[(f32, u32)]) -> Vec<u32> { v.iter().map(|x| x.1).collect() }

fn recall(base: &[u32], cand: &[u32]) -> f32 {
    let bs: std::collections::HashSet<u32> = base.iter().copied().collect();
    let hit = cand.iter().filter(|c| bs.contains(c)).count();
    hit as f32 / base.len() as f32
}

fn main() {
    let n = 50_000usize;
    let d = 128usize;
    let k = 10usize;
    let (h, queries) = make(n, d, 2026);
    let perm_var = DimensionOrder::VarianceDesc.permutation(&h);
    let p = PdxStore::from_horizontal(&h, perm_var);

    println!("== PDX brute-force micro-benchmark ==");
    println!("n={n}, d={d}, k={k}, queries={}", queries.len());
    println!("horizontal bytes ~ {:.1} MiB", (n * d * 4) as f64 / (1024.0*1024.0));
    println!("PDX bytes         = {:.1} MiB", p.bytes() as f64 / (1024.0*1024.0));

    // Warmup.
    let _ = brute_horizontal(&h, &queries[0], k);
    let _ = brute_pdx(&p, &queries[0], k);
    let _ = brute_pdx_pruned(&p, &queries[0], k, 8);

    // --- Horizontal baseline ---
    let t = Instant::now();
    let mut base_ids: Vec<Vec<u32>> = Vec::with_capacity(queries.len());
    for q in &queries {
        base_ids.push(ids_of(&brute_horizontal(&h, q, k)));
    }
    let horiz_ms = t.elapsed().as_secs_f64() * 1000.0;
    let horiz_per = horiz_ms / queries.len() as f64;

    // --- PDX plain ---
    let t = Instant::now();
    let mut pdx_recall_sum = 0f32;
    for (i, q) in queries.iter().enumerate() {
        let r = ids_of(&brute_pdx(&p, q, k));
        pdx_recall_sum += recall(&base_ids[i], &r);
    }
    let pdx_ms = t.elapsed().as_secs_f64() * 1000.0;
    let pdx_per = pdx_ms / queries.len() as f64;
    let pdx_recall = pdx_recall_sum / queries.len() as f32;

    // --- PDX pruned ---
    let t = Instant::now();
    let mut pr_recall_sum = 0f32;
    let mut work_ratio_sum = 0f64;
    let mut early_blocks = 0u64;
    let mut total_blocks = 0u64;
    for (i, q) in queries.iter().enumerate() {
        let (r, s) = brute_pdx_pruned(&p, q, k, 8);
        pr_recall_sum += recall(&base_ids[i], &ids_of(&r));
        work_ratio_sum += s.work_ratio();
        early_blocks += s.blocks_early_exit;
        total_blocks += s.blocks_total;
    }
    let pr_ms = t.elapsed().as_secs_f64() * 1000.0;
    let pr_per = pr_ms / queries.len() as f64;
    let pr_recall = pr_recall_sum / queries.len() as f32;
    let avg_work = work_ratio_sum / queries.len() as f64;

    println!();
    println!("Brute-force (100 queries):");
    println!("  horizontal:  {:8.2} ms total   {:6.3} ms/query   recall 1.000 (baseline)",
             horiz_ms, horiz_per);
    println!("  pdx plain:   {:8.2} ms total   {:6.3} ms/query   recall {:.3}   speedup {:.2}x",
             pdx_ms, pdx_per, pdx_recall, horiz_per / pdx_per);
    println!("  pdx pruned:  {:8.2} ms total   {:6.3} ms/query   recall {:.3}   speedup {:.2}x",
             pr_ms, pr_per, pr_recall, horiz_per / pr_per);
    println!("  avg work ratio (visited/total cells): {:.3}", avg_work);
    println!("  blocks early-exit: {} / {} ({:.1}%)",
             early_blocks, total_blocks, 100.0 * early_blocks as f64 / total_blocks as f64);

    // --- PDX block-pruned (branchless inner loop) ---
    let _ = brute_pdx_block_pruned(&p, &queries[0], k, 16);
    let t = Instant::now();
    let mut bp_recall_sum = 0f32;
    let mut bp_work_sum = 0f64;
    let mut bp_early = 0u64;
    let mut bp_total = 0u64;
    for (i, q) in queries.iter().enumerate() {
        let (r, s) = brute_pdx_block_pruned(&p, q, k, 16);
        bp_recall_sum += recall(&base_ids[i], &ids_of(&r));
        bp_work_sum += s.work_ratio();
        bp_early += s.blocks_early_exit;
        bp_total += s.blocks_total;
    }
    let bp_ms = t.elapsed().as_secs_f64() * 1000.0;
    let bp_per = bp_ms / queries.len() as f64;
    let bp_recall = bp_recall_sum / queries.len() as f32;
    let bp_work = bp_work_sum / queries.len() as f64;
    println!("  pdx block-prune (dim_step=16, branchless):");
    println!("               {:8.2} ms total   {:6.3} ms/query   recall {:.3}   speedup {:.2}x   work {:.3}   early {}/{}",
             bp_ms, bp_per, bp_recall, horiz_per / bp_per, bp_work, bp_early, bp_total);

    // --- IVF ---
    println!();
    println!("== IVF nprobe study (n_lists=64, iters=8) ==");
    let ivf_h = IvfHorizontal::build(&h, 64, 8, 7);
    let ivf_p = IvfPdxPruned::build(&h, 64, 8, 7, 8);

    for &nprobe in &[4usize, 8, 16] {
        let t = Instant::now();
        let mut hits_sum = 0f32;
        let mut cand_sum = 0u64;
        for (i, q) in queries.iter().enumerate() {
            let (r, s) = ivf_h.search(q, k, nprobe);
            hits_sum += recall(&base_ids[i], &ids_of(&r));
            cand_sum += s.candidates;
        }
        let ms_h = t.elapsed().as_secs_f64() * 1000.0;
        let per_h = ms_h / queries.len() as f64;
        let rec_h = hits_sum / queries.len() as f32;
        let avg_cand = cand_sum as f64 / queries.len() as f64;

        let t = Instant::now();
        let mut hits_sum = 0f32;
        let mut cand_sum = 0u64;
        let mut work_sum = 0f64;
        for (i, q) in queries.iter().enumerate() {
            let (r, s) = ivf_p.search(q, k, nprobe);
            hits_sum += recall(&base_ids[i], &ids_of(&r));
            cand_sum += s.candidates;
            work_sum += s.prune.work_ratio();
        }
        let ms_p = t.elapsed().as_secs_f64() * 1000.0;
        let per_p = ms_p / queries.len() as f64;
        let rec_p = hits_sum / queries.len() as f32;
        let avg_work = work_sum / queries.len() as f64;

        println!("nprobe={nprobe:2}  candidates ~ {:6.0}", avg_cand);
        println!("  ivf horizontal : {:6.3} ms/query   recall {:.3}", per_h, rec_h);
        println!("  ivf pdx-pruned : {:6.3} ms/query   recall {:.3}   work {:.3}   speedup {:.2}x",
                 per_p, rec_p, avg_work, per_h / per_p);
        let _ = avg_cand;
    }

    // --- Dimension-order ablation on brute-force PDX ---
    println!();
    println!("== Dimension-order ablation (brute-force PDX, dim_step=8) ==");
    let p_id = PdxStore::from_horizontal(&h, DimensionOrder::Identity.permutation(&h));
    let p_vd = &p;
    let t = Instant::now();
    let mut sum_w = 0f64;
    for q in &queries { let (_, s) = brute_pdx_pruned(&p_id, q, k, 8); sum_w += s.work_ratio(); }
    let ms_id = t.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
    let w_id = sum_w / queries.len() as f64;
    let t = Instant::now();
    let mut sum_w = 0f64;
    for q in &queries { let (_, s) = brute_pdx_pruned(p_vd, q, k, 8); sum_w += s.work_ratio(); }
    let ms_vd = t.elapsed().as_secs_f64() * 1000.0 / queries.len() as f64;
    let w_vd = sum_w / queries.len() as f64;
    println!("  identity order       : {:6.3} ms/query   work {:.3}", ms_id, w_id);
    println!("  variance-desc order  : {:6.3} ms/query   work {:.3}", ms_vd, w_vd);
}
