use ruvector_ef_hnsw_adjacency::{
    bfs_touch, build_random_graph, ef::EFAdjacency, fill_from, packed::PackedAdjacency,
    plain::PlainAdjacency, Adjacency,
};

fn report<A: Adjacency>(name: &str, adj: &A, baseline_bytes: usize) {
    let b = adj.bytes();
    let ratio = baseline_bytes as f64 / b as f64;
    println!(
        "  {:<8} bytes = {:>10}  ({:.2}x vs plain)",
        name, b, ratio
    );
}

fn main() {
    let configs = [
        (10_000usize, 16usize),
        (10_000, 32),
        (100_000, 16),
        (100_000, 32),
    ];
    for (n, m) in configs {
        println!("\n== n = {}, m = {} ==", n, m);
        let g = build_random_graph(n, m, 42);
        let mut plain = PlainAdjacency::new();
        let mut packed = PackedAdjacency::new(n);
        let mut ef = EFAdjacency::new(n);
        fill_from(&mut plain, &g);
        fill_from(&mut packed, &g);
        fill_from(&mut ef, &g);

        let base = plain.bytes();
        report("plain", &plain, base);
        report("packed", &packed, base);
        report("ef", &ef, base);

        // Sanity: BFS edge counts match.
        let a = bfs_touch(&plain, 0, 128);
        let b = bfs_touch(&packed, 0, 128);
        let c = bfs_touch(&ef, 0, 128);
        assert_eq!(a, b);
        assert_eq!(b, c);
        println!("  bfs touched edges = {}", a);
    }
}
