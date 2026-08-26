//! Quickstart: build 100k-node synthetic HNSW layer with three adjacency
//! backends, print the memory savings, and run one decode.

use ruvector_delta_varint_hnsw_adjacency::*;

fn main() {
    let n = 100_000;
    let m = 32;
    println!("Generating {n} nodes with M={m} (high locality)...");
    let nodes = synth_neighbors(n, m, 0.95, 0xDEADBEEF_CAFE_BABE);

    let plain = PlainAdjacency::build(nodes.clone(), m);
    let dv = DeltaVarintAdjacency::build(nodes.clone(), m);
    let pfor = PforBlockedAdjacency::build(nodes.clone(), m);

    println!("{}", footprint("plain-u32", &plain));
    println!("{}", footprint("delta-varint", &dv));
    println!("{}", footprint("pfor-blocked", &pfor));

    let mut buf = vec![0u32; m];
    let node = 42;
    let n_dec = pfor.decode_into(node, &mut buf);
    println!("\nNeighbors of node {node} (pfor-blocked): {:?}", &buf[..n_dec]);
}
