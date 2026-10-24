//! Minimal harness-free micro-bench. Running on a small graph so
//! `cargo bench` finishes quickly; the heavier numbers come from the
//! `examples/bench.rs` binary.

use std::time::Instant;

use ruvector_finger_hnsw::*;

fn main() {
    let n = 1500usize;
    let d = 96usize;
    let m = 24usize;
    let n_q = 50usize;
    let k = 10usize;
    let ef = 48usize;
    let sketch_k = 16usize;

    let data = gen_dataset(n, d, 16, 0xBEEF);
    let queries = gen_queries_from_data(&data, n_q, 0.5, 0xCAFE);
    let graph = FlatGraph::build(&data, &GraphParams { m, m_long: 4, seed: 0xC0FFEE });
    let sketch = EdgeSketch::build(&data, &graph, &SketchParams { k: sketch_k, seed: 0xA17EC0DE });

    let variants: Vec<(&str, Box<dyn Fn(&[f32]) -> SearchResult>)> = vec![
        ("exact", Box::new(|q: &[f32]| ExactSearch.search(&data, &graph, q, k, ef))),
        ("finger_pure", Box::new(|q: &[f32]| FingerSearch { sketch: &sketch }.search(&data, &graph, q, k, ef))),
        ("finger_guarded", Box::new(|q: &[f32]| FingerGuarded { sketch: &sketch, slack: 0.20 }.search(&data, &graph, q, k, ef))),
    ];

    for (label, f) in variants {
        let t = Instant::now();
        let mut acc = 0u64;
        for q in &queries {
            let r = f(q);
            acc = acc.wrapping_add(r.stats.exact_dist_calls);
        }
        let elapsed = t.elapsed().as_secs_f32();
        println!("{:>16}  {:>7.3}s  {} total exact L2 calls", label, elapsed, acc);
    }
}
