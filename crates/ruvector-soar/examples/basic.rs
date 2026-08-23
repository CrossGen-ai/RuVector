//! Smallest possible SOAR usage example.
use ruvector_soar::{kmeans, IvfSoar, PartitionIndex, Vector, rng::Xor64};

fn main() {
    let mut rng = Xor64::new(1);
    let raw: Vec<Vec<f32>> = (0..500)
        .map(|i| {
            let c = (i % 8) as f32;
            (0..16).map(|_| c + rng.gauss() * 0.4).collect()
        })
        .collect();
    let vecs: Vec<Vector> = raw.iter().enumerate()
        .map(|(i, v)| Vector { id: i as u32, data: v.clone() })
        .collect();
    let cents = kmeans::train(&raw, 16, 8, 42);
    let ix = IvfSoar::build(vecs, cents, 1.5);
    let q = vec![3f32; 16];
    let hits = ix.search(&q, 5, 3);
    println!("SOAR built: {} partitions, {} bytes postings", ix.nlist(), ix.posting_bytes());
    for (d, id) in hits { println!("  id={id} d²={d:.3}"); }
}
