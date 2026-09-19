//! Minimal `cargo run --example quickstart` demonstration.
use ruvector_deg_frozen_backbone::{AnnIndex, FrozenBackboneDeg, make_clustered_dataset};

fn main() {
    let dim = 32;
    let mut idx = FrozenBackboneDeg::new(dim, 12, 48, 0.25);
    let data = make_clustered_dataset(1000, dim, 8, 1);
    for (i, v) in data.iter().enumerate() {
        idx.insert(i as u32, v.clone());
    }
    let q = &data[7];
    let hits = idx.search(q, 5, 48);
    println!("frozen-backbone DEG returned {} hits for a training-set query:", hits.len());
    for (id, d) in &hits {
        println!("  id={id:>4}  d^2={d:.4}");
    }
    println!("edges total: {}", idx.n_edges());
}
