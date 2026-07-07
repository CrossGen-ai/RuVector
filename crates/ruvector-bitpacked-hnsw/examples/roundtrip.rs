//! Minimal example: build a random graph and check that all three backends
//! decode the same neighbor sets.

use ruvector_bitpacked_hnsw::{
    BitPackedStore, DeltaVarintStore, NeighborStore, RawU32Store,
};

fn main() {
    let n = 2_000usize;
    let m = 16usize;
    let mut s: u64 = 0xB1_7_9AC5;
    let mut step = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };
    let g: Vec<Vec<u32>> = (0..n)
        .map(|_| (0..m).map(|_| (step() as u32) % (n as u32)).collect())
        .collect();

    let raw = RawU32Store::build(&g);
    let vb = DeltaVarintStore::build(&g);
    let bp = BitPackedStore::build(&g);

    let edges = (n * m) as f64;
    println!("raw:         {} bytes  ({:.2} bpe)", raw.bytes(), raw.bytes() as f64 / edges);
    println!("delta-varint: {} bytes  ({:.2} bpe)", vb.bytes(), vb.bytes() as f64 / edges);
    println!("bit-packed:  {} bytes  ({:.2} bpe)", bp.bytes(), bp.bytes() as f64 / edges);

    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    for i in 0..n as u32 {
        raw.decode(i, &mut a);
        vb.decode(i, &mut b);
        bp.decode(i, &mut c);
        let mut a_sorted = a.clone();
        a_sorted.sort_unstable();
        assert_eq!(a_sorted, b);
        assert_eq!(b, c);
    }
    println!("OK — all {n} nodes decode to identical sorted neighbor sets across backends.");
}
