//! `cargo run --release --bin drb-bench` — real benchmark, real numbers.
//!
//! Prints a markdown table ready for the research doc / gist.

use ruvector_dyn_rerank_budget::*;

fn main() {
    let dim = 128usize;
    let n = 20_000usize;
    let nq_cal = 200usize;
    let nq_eval = 1_000usize;
    let topk = 10usize;
    let nprobe = 6usize;
    let cand_cap = 800usize;
    let pq_scale = 0.8f32;
    let n_cells = 16usize;

    eprintln!("[drb-bench] generating {} vectors × {}d", n, dim);
    let data = make_dataset(n, dim, 1);

    eprintln!("[drb-bench] generating {} calibration + {} eval queries",
        nq_cal, nq_eval);
    let cal_queries = make_dataset(nq_cal, dim, 101);
    let eval_queries = make_dataset(nq_eval, dim, 202);

    eprintln!("[drb-bench] training IVF (K={}, 5 iters)", n_cells);
    let t0 = std::time::Instant::now();
    let ivf = Ivf::train(&data, n_cells, 5, 7);
    eprintln!("    done in {:.1} ms", t0.elapsed().as_secs_f32() * 1000.0);

    eprintln!("[drb-bench] computing exact top-{} for eval set", topk);
    let t0 = std::time::Instant::now();
    let truths: Vec<Vec<u32>> = eval_queries.iter()
        .map(|q| exact_topk(&data, q, topk)).collect();
    eprintln!("    done in {:.1} ms", t0.elapsed().as_secs_f32() * 1000.0);

    eprintln!("[drb-bench] calibrating gap_ref over {} queries", nq_cal);
    let gap_ref = calibrate_gap_ref(
        &ivf, &data, &cal_queries, topk, nprobe, cand_cap, pq_scale, 32,
    );
    eprintln!("    gap_ref = {:.4}", gap_ref);

    let runs = vec![
        run_policy(&ivf, &data, &eval_queries, &truths, topk, nprobe, cand_cap, pq_scale,
            Box::new(Fixed(25))),
        run_policy(&ivf, &data, &eval_queries, &truths, topk, nprobe, cand_cap, pq_scale,
            Box::new(Fixed(50))),
        run_policy(&ivf, &data, &eval_queries, &truths, topk, nprobe, cand_cap, pq_scale,
            Box::new(Fixed(100))),
        run_policy(&ivf, &data, &eval_queries, &truths, topk, nprobe, cand_cap, pq_scale,
            Box::new(Fixed(200))),
        run_policy(&ivf, &data, &eval_queries, &truths, topk, nprobe, cand_cap, pq_scale,
            Box::new(DrbGap { r_min: 25, r_max: 200, probe: 32, gap_ref })),
        run_policy(&ivf, &data, &eval_queries, &truths, topk, nprobe, cand_cap, pq_scale,
            Box::new(DrbGap { r_min: 15, r_max: 150, probe: 32, gap_ref })),
    ];

    // Oracle pass (reports mean & p95 R only; recall ~= 1 by definition).
    let mut oracle_rs: Vec<f32> = Vec::with_capacity(eval_queries.len());
    let t0 = std::time::Instant::now();
    for (q, truth) in eval_queries.iter().zip(truths.iter()) {
        let cands = coarse_scan(&ivf, &data, q, nprobe, cand_cap, pq_scale);
        let r = (Oracle { data: &data, query: q, truth }).choose(&cands, topk);
        oracle_rs.push(r as f32);
    }
    let oracle_ms = t0.elapsed().as_secs_f32() * 1000.0;
    oracle_rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let oracle_mean = oracle_rs.iter().sum::<f32>() / oracle_rs.len() as f32;
    let oracle_p95 = oracle_rs[(oracle_rs.len() as f32 * 0.95) as usize];

    println!("\n## DRB benchmark ({} × {}d, nq={}, k={}, nprobe={})", n, dim, nq_eval, topk, nprobe);
    println!("| policy | recall@{} | mean R | p95 R | wall-ms |", topk);
    println!("|---|---|---|---|---|");
    for r in &runs {
        println!("| {} | {:.3} | {:.1} | {:.0} | {:.1} |",
            r.name, r.mean_recall, r.mean_r, r.p95_r, r.elapsed_ms);
    }
    println!("| oracle | 1.000 | {:.1} | {:.0} | {:.1} |",
        oracle_mean, oracle_p95, oracle_ms);
}

fn run_policy(
    ivf: &Ivf, data: &[VecF32], queries: &[VecF32], truths: &[Vec<u32>],
    topk: usize, nprobe: usize, cand_cap: usize, pq_scale: f32,
    policy: Box<dyn BudgetPolicy>,
) -> RunResult {
    evaluate(ivf, data, queries, truths, topk, nprobe, cand_cap, pq_scale, &*policy)
}
