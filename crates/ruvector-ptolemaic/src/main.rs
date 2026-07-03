//! `ptolemaic-demo` — runnable comparison of Linear vs Triangle vs Ptolemaic
//! pivot pruning on a small deterministic dataset. Prints DCOs and prune rate.

use ruvector_ptolemaic::{
    gen_dataset, gen_queries,
    pivot::select_farthest_first,
    KnnIndex, LinearScan, PtolemaicPivot, TrianglePivot,
};
use std::time::Instant;

fn main() {
    let n = 5_000usize;
    let dim = 32usize;
    let clusters = 12usize;
    let k = 10usize;
    let n_queries = 100usize;
    let p = 8usize;

    println!("== ruvector-ptolemaic demo ==");
    println!("n={n} dim={dim} k={k} queries={n_queries} pivots={p}");

    let data = gen_dataset(n, dim, clusters, 42);
    let queries = gen_queries(n_queries, dim, 7);
    let pivots = select_farthest_first(&data, dim, p, 123);

    let linear = LinearScan::new(data.clone(), dim).unwrap();
    let tri = TrianglePivot::build(data.clone(), dim, pivots.clone()).unwrap();
    let pto = PtolemaicPivot::build(data.clone(), dim, pivots).unwrap();

    let mut lin_dcos = 0u64;
    let mut tri_dcos = 0u64;
    let mut pto_dcos = 0u64;
    let mut tri_pruned = 0u64;
    let mut pto_pruned = 0u64;

    let t0 = Instant::now();
    for q in 0..n_queries {
        let qv = &queries[q * dim..(q + 1) * dim];
        let (_, s) = linear.search(qv, k).unwrap();
        lin_dcos += s.dcos;
    }
    let t_lin = t0.elapsed();

    let t0 = Instant::now();
    for q in 0..n_queries {
        let qv = &queries[q * dim..(q + 1) * dim];
        let (_, s) = tri.search(qv, k).unwrap();
        tri_dcos += s.dcos;
        tri_pruned += s.pruned;
    }
    let t_tri = t0.elapsed();

    let t0 = Instant::now();
    for q in 0..n_queries {
        let qv = &queries[q * dim..(q + 1) * dim];
        let (_, s) = pto.search(qv, k).unwrap();
        pto_dcos += s.dcos;
        pto_pruned += s.pruned;
    }
    let t_pto = t0.elapsed();

    // Correctness cross-check on query 0.
    let qv = &queries[0..dim];
    let a = linear.search(qv, k).unwrap().0;
    let b = tri.search(qv, k).unwrap().0;
    let c = pto.search(qv, k).unwrap().0;
    let ids_a: Vec<u32> = a.iter().map(|n| n.id).collect();
    let ids_b: Vec<u32> = b.iter().map(|n| n.id).collect();
    let ids_c: Vec<u32> = c.iter().map(|n| n.id).collect();
    assert_eq!(ids_a, ids_b, "Triangle backend differs from Linear");
    assert_eq!(ids_a, ids_c, "Ptolemaic backend differs from Linear");

    println!();
    println!("--- Results ---");
    println!(
        "Linear:    DCOs/query={:>8.1}  time={:>8.2}ms",
        lin_dcos as f64 / n_queries as f64,
        t_lin.as_secs_f64() * 1000.0
    );
    println!(
        "Triangle:  DCOs/query={:>8.1}  pruned/query={:>7.1}  time={:>8.2}ms",
        tri_dcos as f64 / n_queries as f64,
        tri_pruned as f64 / n_queries as f64,
        t_tri.as_secs_f64() * 1000.0
    );
    println!(
        "Ptolemaic: DCOs/query={:>8.1}  pruned/query={:>7.1}  time={:>8.2}ms",
        pto_dcos as f64 / n_queries as f64,
        pto_pruned as f64 / n_queries as f64,
        t_pto.as_secs_f64() * 1000.0
    );
    let baseline = lin_dcos as f64;
    println!(
        "Triangle prune vs baseline:   {:.1}% DCOs cut",
        100.0 * (1.0 - tri_dcos as f64 / baseline)
    );
    println!(
        "Ptolemaic prune vs baseline:  {:.1}% DCOs cut",
        100.0 * (1.0 - pto_dcos as f64 / baseline)
    );
    println!(
        "Ptolemaic vs Triangle:        {:.1}% additional DCOs cut",
        100.0 * (1.0 - pto_dcos as f64 / tri_dcos.max(1) as f64)
    );
    println!();
    println!("All backends returned identical k-NN ordering ✓");
}
