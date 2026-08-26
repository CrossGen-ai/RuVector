use ruvector_delta_varint_hnsw_adjacency::*;

fn assert_roundtrip(nodes: Vec<Vec<u32>>, m: usize) {
    let expected: Vec<Vec<u32>> = nodes
        .iter()
        .map(|ns| {
            let mut s: Vec<u32> = ns.clone();
            s.sort_unstable();
            s.dedup();
            s.truncate(m);
            s
        })
        .collect();

    let plain = PlainAdjacency::build(nodes.clone(), m);
    let dv = DeltaVarintAdjacency::build(nodes.clone(), m);
    let pfor = PforBlockedAdjacency::build(nodes.clone(), m);

    let mut buf = vec![0u32; m];
    for i in 0..expected.len() as u32 {
        let n1 = plain.decode_into(i, &mut buf);
        assert_eq!(&buf[..n1], expected[i as usize].as_slice(), "plain node {}", i);
        let n2 = dv.decode_into(i, &mut buf);
        assert_eq!(&buf[..n2], expected[i as usize].as_slice(), "delta-varint node {}", i);
        let n3 = pfor.decode_into(i, &mut buf);
        assert_eq!(&buf[..n3], expected[i as usize].as_slice(), "pfor node {}", i);
    }
}

#[test]
fn empty_graph() {
    assert_roundtrip(vec![], 16);
}

#[test]
fn single_node_no_neighbors() {
    assert_roundtrip(vec![vec![]], 16);
}

#[test]
fn dense_local_graph() {
    let m = 32;
    let n = 1_000;
    let nodes = synth_neighbors(n, m, 0.95, 0xDEADBEEF);
    assert_roundtrip(nodes, m);
}

#[test]
fn uniform_random_graph() {
    let m = 16;
    let n = 5_000;
    let nodes = synth_neighbors(n, m, 0.0, 0x1234_5678_9ABC_DEF0);
    assert_roundtrip(nodes, m);
}

#[test]
fn handles_duplicates_and_over_m() {
    let m = 4;
    let nodes = vec![vec![7, 3, 3, 1, 9, 2, 2, 8]];
    // Normalized sorted = [1,2,3,7,8,9]; truncated to m=4 = [1,2,3,7]
    let plain = PlainAdjacency::build(nodes.clone(), m);
    let dv = DeltaVarintAdjacency::build(nodes.clone(), m);
    let pfor = PforBlockedAdjacency::build(nodes.clone(), m);
    let mut buf = vec![0u32; m];
    assert_eq!(plain.decode_into(0, &mut buf), 4);
    assert_eq!(&buf[..4], &[1, 2, 3, 7]);
    assert_eq!(dv.decode_into(0, &mut buf), 4);
    assert_eq!(&buf[..4], &[1, 2, 3, 7]);
    assert_eq!(pfor.decode_into(0, &mut buf), 4);
    assert_eq!(&buf[..4], &[1, 2, 3, 7]);
}

#[test]
fn footprint_shrinks_for_local_graph() {
    let m = 32;
    let n = 5_000;
    let nodes = synth_neighbors(n, m, 0.98, 0xCAFEBABE);
    let plain = PlainAdjacency::build(nodes.clone(), m);
    let dv = DeltaVarintAdjacency::build(nodes.clone(), m);
    let pfor = PforBlockedAdjacency::build(nodes.clone(), m);
    let bp = plain.bytes();
    let bd = dv.bytes();
    let bpf = pfor.bytes();
    // For a highly local graph, both compressed forms must beat plain.
    assert!(bd < bp, "delta-varint should shrink plain: {} !< {}", bd, bp);
    assert!(bpf < bp, "pfor should shrink plain: {} !< {}", bpf, bp);
}
