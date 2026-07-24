//! Small bench binary (no criterion dep). Runs a tiny per-query timing
//! loop over each variant. Used mainly to validate that the build path
//! compiles as a bench target; the real numbers come from `cargo run
//! --release --example muvera_bench`.
use ruvector_muvera::{
    chamfer::l2_normalize_set,
    fde::FdeParams,
    ivf::IvfParams,
    retriever::{Document, FlatMaxSim, MultiVectorRetriever, MuveraFlat},
    MuveraIvf,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn main() {
    let d = 16;
    let n = 500;
    let tok = 8;
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let corpus: Vec<Document> = (0..n)
        .map(|id| {
            let mut t = vec![0.0f32; tok * d];
            for x in t.iter_mut() {
                let u1: f32 = rng.gen::<f32>().max(1e-9);
                let u2: f32 = rng.gen::<f32>();
                *x = (-2.0f32 * u1.ln()).sqrt()
                    * (2.0f32 * std::f32::consts::PI * u2).cos();
            }
            l2_normalize_set(&mut t, d);
            Document { id: id as u32, tokens: t, n_tokens: tok }
        })
        .collect();
    let query = corpus[0].tokens.clone();

    let mut flat = FlatMaxSim::new(d);
    let mut mf = MuveraFlat::new(FdeParams { d, k_sim: 4, reps: 8, seed: 1 });
    let mut ivf = MuveraIvf::new(
        FdeParams { d, k_sim: 4, reps: 8, seed: 1 },
        IvfParams { n_lists: 8, n_probe: 4, candidates: 32, rerank: 8, kmeans_iters: 5, seed: 2 },
    );
    flat.build(&corpus);
    mf.build(&corpus);
    ivf.build(&corpus);

    for name in &["flat", "muvera_flat", "muvera_ivf"] {
        let t = Instant::now();
        for _ in 0..100 {
            let hits = match *name {
                "flat" => flat.search(&query, 5),
                "muvera_flat" => mf.search(&query, 5),
                _ => ivf.search(&query, 5),
            };
            std::hint::black_box(hits);
        }
        let us = t.elapsed().as_micros();
        println!("{:>12}: {} us total / 100 queries = {} us/q", name, us, us / 100);
    }
}
