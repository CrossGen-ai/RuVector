//! Real benchmark: baseline FlatIvf vs greedy-chain-reordered vs projection-sort-reordered.
//!
//! No mocks. Numbers come from `cargo run --release --bin bench`.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use ruvector_ivf_centroid_locality_reorder::{
    apply_permutation, estimated_bytes, probe_stride, FlatIvf, GreedyChain, ProjectionSort,
    Reorder,
};
use std::time::Instant;

fn gen(seed: u64, n: usize, dim: usize) -> Vec<f32> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.gen::<f32>() - 0.5).collect()
}

fn bench(ivf: &FlatIvf, queries: &[f32], dim: usize, nprobe: usize, k: usize) -> (f64, i64) {
    let nq = queries.len() / dim;
    // Warmup.
    for qi in 0..nq.min(16) {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let _ = ivf.search_topk(q, nprobe, k);
    }
    // Timed.
    let t = Instant::now();
    let mut sink: u64 = 0;
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let hits = ivf.search_topk(q, nprobe, k);
        sink = sink.wrapping_add(hits[0].1 as u64);
    }
    let elapsed = t.elapsed().as_secs_f64();
    // Side-effect to defeat DCE.
    std::hint::black_box(sink);

    let mut stride_sum = 0i64;
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        stride_sum += probe_stride(ivf, q, nprobe);
    }
    let avg_stride = stride_sum / nq as i64;
    (elapsed * 1e6 / nq as f64, avg_stride)
}

fn main() {
    let dim = 128;
    let n = 50_000;
    let n_clusters = 512;
    let nq = 1_000;
    let nprobe = 24;
    let k = 10;

    println!("# IVF Centroid Locality Reorder — benchmark");
    println!(
        "dim={dim}, n_points={n}, n_clusters={n_clusters}, nq={nq}, nprobe={nprobe}, k={k}"
    );
    println!();

    let data = gen(1, n, dim);
    let queries = gen(2, nq, dim);

    let t = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let ivf = FlatIvf::new(&mut rng, &data, dim, n_clusters, 8);
    println!("Built baseline IVF in {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let perm_greedy = GreedyChain::permutation(&ivf.centroids, ivf.dim, ivf.n_clusters);
    let dt_greedy = t.elapsed().as_secs_f64();
    let ivf_greedy = apply_permutation(&ivf, &perm_greedy);
    println!("Greedy chain permutation computed in {:.4}s", dt_greedy);

    let t = Instant::now();
    let perm_proj = ProjectionSort::permutation(&ivf.centroids, ivf.dim, ivf.n_clusters);
    let dt_proj = t.elapsed().as_secs_f64();
    let ivf_proj = apply_permutation(&ivf, &perm_proj);
    println!("Projection-sort permutation computed in {:.4}s", dt_proj);
    println!();

    let bytes = estimated_bytes(&ivf);
    println!(
        "Memory (identical across variants): {:.2} MiB",
        bytes as f64 / (1024.0 * 1024.0)
    );
    println!();

    let (b_us, b_stride) = bench(&ivf, &queries, dim, nprobe, k);
    let (g_us, g_stride) = bench(&ivf_greedy, &queries, dim, nprobe, k);
    let (p_us, p_stride) = bench(&ivf_proj, &queries, dim, nprobe, k);

    println!("Variant             |  µs / query | avg probe stride");
    println!("--------------------+-------------+------------------");
    println!("baseline            |  {:9.2}  |  {:>14}", b_us, b_stride);
    println!("greedy-chain        |  {:9.2}  |  {:>14}", g_us, g_stride);
    println!("projection-sort     |  {:9.2}  |  {:>14}", p_us, p_stride);
    println!();
    println!(
        "Speedup greedy vs baseline:  {:.2}x",
        b_us / g_us
    );
    println!(
        "Speedup projection vs baseline: {:.2}x",
        b_us / p_us
    );
    println!(
        "Stride reduction greedy:     {:.1}%",
        100.0 * (1.0 - g_stride as f64 / b_stride as f64)
    );
    println!(
        "Stride reduction projection: {:.1}%",
        100.0 * (1.0 - p_stride as f64 / b_stride as f64)
    );
}
