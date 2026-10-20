use ruvector_pivot_tri_ivf::{gen_dataset, IvfIndex};

fn main() {
    let data = gen_dataset(2000, 64, 1);
    let idx = IvfIndex::build(&data, 32, 6, 8, 1);
    let q = gen_dataset(1, 64, 42).remove(0);
    let (hits, stats) = idx.search(&q, 10, 8, true);
    println!("top-10 for sample query:");
    for h in &hits {
        println!("  id={:>6}  d2={:.4}", h.id, h.dist_sq);
    }
    println!("stats: {:?}", stats);
}
