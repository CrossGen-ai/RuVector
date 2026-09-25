//! Neighbor-store benchmark harness.
//!
//! Runs three backends (Raw, DeltaVarint, EliasFano) over an
//! HNSW-shaped adjacency workload and prints memory + decode-time
//! numbers.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_ef_neighbors::{
    DeltaVarintStore, EliasFanoStore, NeighborStore, RawStore,
};
use std::collections::HashSet;
use std::time::Instant;

fn random_list(rng: &mut StdRng, m: usize, universe: u32) -> Vec<u32> {
    let mut set: HashSet<u32> = HashSet::new();
    while set.len() < m {
        set.insert(rng.gen_range(0..universe));
    }
    set.into_iter().collect()
}

struct Row {
    name: &'static str,
    bytes: usize,
    bits_per_neighbor: f64,
    decode_ns_per_list: f64,
    nth_ns_per_call: f64,
}

fn run_backend<S: NeighborStore>(
    name: &'static str,
    mut store: S,
    lists: &[Vec<u32>],
    n_neighbors_total: usize,
    decode_iters: usize,
    nth_iters: usize,
) -> Row {
    for l in lists {
        store.push_list(l);
    }
    let bytes = store.bytes();
    let bits_per_neighbor = (bytes as f64 * 8.0) / (n_neighbors_total as f64);

    // Full-list decode benchmark.
    let mut sink: u64 = 0;
    let mut out = Vec::with_capacity(64);
    let t0 = Instant::now();
    for _ in 0..decode_iters {
        for i in 0..lists.len() {
            store.get_list(i, &mut out);
            sink = sink.wrapping_add(out.len() as u64);
            sink = sink.wrapping_add(u64::from(out[0]));
        }
    }
    let dt = t0.elapsed().as_nanos() as f64;
    let decode_ns_per_list = dt / (decode_iters * lists.len()) as f64;

    // Random nth-element benchmark (HNSW candidate expansion pattern).
    let mut rng = StdRng::seed_from_u64(999);
    let list_len = lists[0].len();
    let t0 = Instant::now();
    for _ in 0..nth_iters {
        let i = rng.gen_range(0..lists.len());
        let k = rng.gen_range(0..list_len);
        sink = sink.wrapping_add(u64::from(store.nth(i, k)));
    }
    let dt = t0.elapsed().as_nanos() as f64;
    let nth_ns_per_call = dt / nth_iters as f64;

    // Prevent optimizer eliminating the workload.
    std::hint::black_box(sink);

    Row {
        name,
        bytes,
        bits_per_neighbor,
        decode_ns_per_list,
        nth_ns_per_call,
    }
}

fn main() {
    println!("ruvector-ef-neighbors benchmark");
    println!("================================\n");

    // Realistic HNSW top-layer shape: N=200_000 nodes, M=32 neighbors,
    // universe=200_000.
    let n_nodes: usize = 200_000;
    let m: usize = 32;
    let universe: u32 = 200_000;
    println!(
        "Workload: N={n_nodes} nodes, M={m} neighbors each, universe=[0,{universe})"
    );
    println!("Total neighbors stored: {}\n", n_nodes * m);

    let mut rng = StdRng::seed_from_u64(42);
    let lists: Vec<Vec<u32>> =
        (0..n_nodes).map(|_| random_list(&mut rng, m, universe)).collect();
    let total_neighbors = n_nodes * m;

    let raw_row =
        run_backend("RawStore (Vec<u32>)", RawStore::new(), &lists, total_neighbors, 3, 1_000_000);
    let dv_row = run_backend(
        "DeltaVarintStore",
        DeltaVarintStore::new(),
        &lists,
        total_neighbors,
        3,
        1_000_000,
    );
    let ef_row = run_backend(
        "EliasFanoStore",
        EliasFanoStore::new(),
        &lists,
        total_neighbors,
        3,
        1_000_000,
    );

    // Info-theoretic lower bound for a strictly increasing sequence
    // of length m in [0, U): ceil(log2(C(U,m))) / m bits per neighbor.
    // Approximation: log2(U*e/m) bits per element.
    let approx_lower_bound =
        ((f64::from(universe) * std::f64::consts::E) / (m as f64)).log2();

    let rows = [&raw_row, &dv_row, &ef_row];

    println!(
        "{:<24} {:>14} {:>18} {:>18} {:>20}",
        "backend",
        "total bytes",
        "bits/neighbor",
        "ns/list decode",
        "ns/nth (random)"
    );
    println!("{:-<98}", "");
    for r in rows {
        println!(
            "{:<24} {:>14} {:>18.2} {:>18.1} {:>20.1}",
            r.name, r.bytes, r.bits_per_neighbor, r.decode_ns_per_list, r.nth_ns_per_call
        );
    }
    println!();
    println!(
        "Info-theoretic lower bound (approx): {:.2} bits/neighbor",
        approx_lower_bound
    );
    println!(
        "EF vs raw:            {:.2}x smaller",
        raw_row.bytes as f64 / ef_row.bytes as f64
    );
    println!(
        "EF vs delta-varint:  {:.2}x smaller",
        dv_row.bytes as f64 / ef_row.bytes as f64
    );
    println!(
        "EF vs lower bound:   +{:.2} bits/neighbor overhead",
        ef_row.bits_per_neighbor - approx_lower_bound
    );

    // A second workload: dense universe (M/U high) — EF sinks below raw
    // less dramatically. Sanity check.
    println!("\nSecondary workload: N={n_nodes}, M={m}, universe=[0,{})", m * 4);
    let mut rng2 = StdRng::seed_from_u64(43);
    let lists2: Vec<Vec<u32>> =
        (0..n_nodes).map(|_| random_list(&mut rng2, m, (m * 4) as u32)).collect();
    let raw2 = run_backend("RawStore", RawStore::new(), &lists2, total_neighbors, 3, 200_000);
    let dv2 =
        run_backend("DeltaVarintStore", DeltaVarintStore::new(), &lists2, total_neighbors, 3, 200_000);
    let ef2 =
        run_backend("EliasFanoStore", EliasFanoStore::new(), &lists2, total_neighbors, 3, 200_000);
    for r in [&raw2, &dv2, &ef2] {
        println!(
            "{:<24} {:>14} bytes  {:>7.2} bits/neigh",
            r.name, r.bytes, r.bits_per_neighbor
        );
    }
}
