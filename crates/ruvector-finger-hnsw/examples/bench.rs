//! Benchmark binary. Produces the real numbers quoted in the research doc.
//!
//! Measures three variants on the same graph and query set:
//!   - Exact (baseline)
//!   - FingerSearch (pure sketch + rerank)
//!   - FingerGuarded (sketch as lower-bound filter)
//!
//! Reports: wall-clock per query, exact L2 computations per query, recall@10
//! against brute-force ground truth.

use std::time::Instant;

use ruvector_finger_hnsw::*;

fn main() {
    // Scale chosen so the whole run finishes in a few seconds on a laptop
    // while giving meaningful separation between variants.
    // Allow overriding d from env so we can show how the sketch's
    // relative win grows with ambient dimension (exact L2 is O(d) per
    // call, sketch inner product stays O(sketch_k)).
    let n = env_usize("FINGER_N", 4000);
    let d = env_usize("FINGER_D", 256);
    let n_queries = env_usize("FINGER_Q", 200);
    let m = env_usize("FINGER_M", 32);
    let m_long = env_usize("FINGER_MLONG", 6);
    let k = env_usize("FINGER_K", 10);
    let ef = env_usize("FINGER_EF", 64);
    let sketch_k = env_usize("FINGER_SK", 32);
    let slack = env_f32("FINGER_SLACK", 0.15);

    println!("ruvector-finger-hnsw benchmark");
    println!("  n={}  d={}  m={}  n_queries={}  k={}  ef={}  sketch_k={}  slack={:.2}",
        n, d, m, n_queries, k, ef, sketch_k, slack);

    print!("building dataset...");
    let t = Instant::now();
    let data = gen_dataset(n, d, 32, 0xA11CE);
    let queries = gen_queries_from_data(&data, n_queries, 0.5, 0xB0B);
    println!(" {:.2}s", t.elapsed().as_secs_f32());

    print!("building graph (brute-force kNN, m={})...", m);
    let t = Instant::now();
    let graph = FlatGraph::build(&data, &GraphParams { m, m_long, seed: 0xC0FFEE });
    println!(" {:.2}s", t.elapsed().as_secs_f32());

    print!("building edge sketch (rank={})...", sketch_k);
    let t = Instant::now();
    let sketch = EdgeSketch::build(&data, &graph, &SketchParams { k: sketch_k, seed: 0xA17EC0DE });
    println!(" {:.2}s", t.elapsed().as_secs_f32());

    println!("sketch memory: {} bytes ({:.2} MB)",
        sketch.bytes, sketch.bytes as f64 / (1024.0 * 1024.0));

    print!("computing ground truth (brute-force) for {} queries...", n_queries);
    let t = Instant::now();
    let gt: Vec<Vec<u32>> = queries.iter()
        .map(|q| brute_force_topk(&data, q, k).iter().map(|p| p.0).collect())
        .collect();
    println!(" {:.2}s", t.elapsed().as_secs_f32());

    // -------- Variant A: exact best-first --------
    let (a_ids, a_dist_calls, a_elapsed) = run("exact", || {
        let s = ExactSearch;
        run_queries(&s, &data, &graph, &queries, k, ef)
    });

    // -------- Variant B: pure sketch + exact rerank --------
    let (b_ids, b_dist_calls, b_elapsed) = run("finger_pure", || {
        let s = FingerSearch { sketch: &sketch };
        run_queries(&s, &data, &graph, &queries, k, ef)
    });

    // -------- Variant C: sketch as lower-bound guard --------
    let (c_ids, c_dist_calls, c_elapsed) = run("finger_guarded", || {
        let s = FingerGuarded { sketch: &sketch, slack };
        run_queries(&s, &data, &graph, &queries, k, ef)
    });

    let a_rec = avg_recall(&gt, &a_ids);
    let b_rec = avg_recall(&gt, &b_ids);
    let c_rec = avg_recall(&gt, &c_ids);

    println!();
    println!("results (lower exact_dist_calls per query is better; higher recall is better):");
    println!();
    println!("  variant         total_s  per_q_ms   exact_dist_calls/q    recall@{}", k);
    println!("  --------------  -------  --------   ------------------    ---------");
    row("exact",          a_elapsed, n_queries, a_dist_calls, a_rec);
    row("finger_pure",    b_elapsed, n_queries, b_dist_calls, b_rec);
    row("finger_guarded", c_elapsed, n_queries, c_dist_calls, c_rec);

    println!();
    let saved = 1.0 - (c_dist_calls as f32 / a_dist_calls as f32);
    println!("finger_guarded saved {:.1}% of exact L2 computations vs exact baseline",
        saved * 100.0);
    println!("finger_pure    saved {:.1}% of exact L2 computations vs exact baseline",
        (1.0 - (b_dist_calls as f32 / a_dist_calls as f32)) * 100.0);

    println!();
    println!("recall deltas: finger_guarded {:+.3}pp vs exact, finger_pure {:+.3}pp vs exact",
        (c_rec - a_rec) * 100.0, (b_rec - a_rec) * 100.0);
}

fn run<F: FnOnce() -> (Vec<Vec<u32>>, u64)>(label: &str, f: F) -> (Vec<Vec<u32>>, u64, f32) {
    print!("running {}...", label);
    let t = Instant::now();
    let (ids, calls) = f();
    let e = t.elapsed().as_secs_f32();
    println!(" {:.2}s  ({} exact L2 calls total)", e, calls);
    (ids, calls, e)
}

fn run_queries<S: Search>(s: &S, data: &[Vec<f32>], graph: &FlatGraph, queries: &[Vec<f32>], k: usize, ef: usize) -> (Vec<Vec<u32>>, u64) {
    let mut ids = Vec::with_capacity(queries.len());
    let mut calls = 0u64;
    for q in queries {
        let r = s.search(data, graph, q, k, ef);
        ids.push(r.ids);
        calls += r.stats.exact_dist_calls;
    }
    (ids, calls)
}

fn avg_recall(gt: &[Vec<u32>], got: &[Vec<u32>]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..gt.len() {
        sum += recall_at_k(&gt[i], &got[i]);
    }
    sum / gt.len() as f32
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn row(label: &str, total_s: f32, n_q: usize, calls: u64, recall: f32) {
    let per_q_ms = total_s * 1000.0 / n_q as f32;
    let per_q_calls = calls as f32 / n_q as f32;
    println!("  {:<14}  {:>7.3}  {:>8.3}   {:>18.1}    {:>8.4}",
        label, total_s, per_q_ms, per_q_calls, recall);
}
