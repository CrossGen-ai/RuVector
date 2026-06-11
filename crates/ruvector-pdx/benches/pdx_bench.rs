//! Custom (no-criterion) benchmark — runs each layout for a fixed wall-clock
//! budget and prints qps. Suitable for `cargo bench -p ruvector-pdx`.

use ruvector_pdx::{synth_corpus, Horizontal, PdxVertical, Scanner};
use std::time::{Duration, Instant};

fn timed<S: Scanner>(s: &S, q: &[f32], k: usize, budget: Duration) -> f64 {
    let _ = s.search(q, k);
    let start = Instant::now();
    let mut runs = 0u64;
    let mut sink = 0u32;
    while start.elapsed() < budget {
        let r = s.search(q, k);
        sink = sink.wrapping_add(r[0].id);
        runs += 1;
    }
    std::hint::black_box(sink);
    runs as f64 / start.elapsed().as_secs_f64()
}

fn main() {
    let configs: &[(usize, usize, usize)] = &[
        (10_000, 128, 10),
        (50_000, 128, 10),
        (10_000, 768, 10),
    ];
    let budget = Duration::from_millis(750);
    println!("# ruvector-pdx benchmark (budget = {:?}/cell)", budget);
    println!("# n,d,k,horizontal_qps,pdx_qps,pdx_pruned_qps");
    for &(n, d, k) in configs {
        let rows = synth_corpus(n, d, 42);
        let q = synth_corpus(1, d, 99).pop().unwrap();
        let h = Horizontal::from_rows(&rows);
        let v = PdxVertical::from_rows(&rows, false);
        let p = PdxVertical::from_rows(&rows, true);
        let hq = timed(&h, &q, k, budget);
        let vq = timed(&v, &q, k, budget);
        let pq = timed(&p, &q, k, budget);
        println!("{n},{d},{k},{hq:.1},{vq:.1},{pq:.1}");
    }
}
