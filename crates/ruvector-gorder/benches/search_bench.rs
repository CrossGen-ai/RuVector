//! Manual (non-criterion) benchmark binary — kept dependency-light so the
//! crate benches without adding criterion to the workspace build.
//! `cargo bench -p ruvector-gorder` runs this in `--release`.

use ruvector_gorder::{
    apply_permutation, search::bench_layout, search::exact_topk, BfsLayout, GorderLayout,
    InsertionLayout, Layout, MiniHnsw, MiniHnswParams,
};

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

fn main() {
    let n = 10_000usize;
    let dim = 64usize;
    let ef = 64usize;
    let k = 10usize;
    let params = MiniHnswParams { dim, m: 16, ef_construction: 64, seed: 42 };
    let g0 = MiniHnsw::build_random(n, params);

    let mut rng = rand::rngs::StdRng::seed_from_u64(0x1234);
    let normal = Normal::new(0.0f32, 1.0f32).unwrap();
    let queries: Vec<Vec<f32>> = (0..200)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rng)).collect();
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= nrm;
            }
            v
        })
        .collect();
    let truth: Vec<Vec<u32>> = queries.iter().map(|q| exact_topk(&g0, q, k)).collect();

    for (l, name) in [
        (Box::new(InsertionLayout::default()) as Box<dyn Layout>, "insertion"),
        (Box::new(BfsLayout::default()) as Box<dyn Layout>, "bfs"),
        (Box::new(GorderLayout { window: 8 }) as Box<dyn Layout>, "gorder"),
    ] {
        let perm = l.permutation(&g0);
        let g = apply_permutation(&g0, &perm);
        let s = bench_layout(&g, &queries, &truth, &perm, ef, k, name);
        println!("{:>10}  qps={:>10.1}  recall={:.3}  visited/q={:.1}", name, s.qps, s.recall_at_k, s.visited_avg);
    }
}
