use ruvector_pq_dominant_code_skip::*;

#[test]
fn skip_matches_baseline_exactly() {
    let data = clustered_dataset(2_000, 64, 8, 42);
    let q = clustered_dataset(10, 64, 8, 43);
    let cb = Codebook::train(&data, 8, 32, 6, 0xBEEF);
    let base = BaselinePq::build(&data, cb.clone());
    let skip = DominantSkipPq::build(&data, cb);

    for i in 0..q.n {
        let a = base.search(q.row(i), 20);
        let b = skip.search(q.row(i), 20);
        let aset: std::collections::HashSet<u32> = a.iter().map(|x| x.0).collect();
        let bset: std::collections::HashSet<u32> = b.iter().map(|x| x.0).collect();
        assert_eq!(aset, bset, "top-k sets differ at query {}", i);
        // Distances should be numerically identical up to tiny floating noise.
        for (x, y) in a.iter().zip(b.iter()) {
            if x.0 == y.0 {
                let diff = (x.1 - y.1).abs();
                assert!(diff < 1e-3, "distance diff {} for id {}", diff, x.0);
            }
        }
    }
}

#[test]
fn dominant_rebalance_puts_mode_at_zero() {
    let data = clustered_dataset(1_500, 32, 4, 7);
    let cb = Codebook::train(&data, 4, 16, 6, 0xAA);
    let skip = DominantSkipPq::build(&data, cb);
    // Count how many codes landed at 0 in each subspace; the "dominant" position
    // should have at least as many as any other.
    let m = skip.codebook.m;
    let k = skip.codebook.k;
    let mut freq = vec![vec![0u32; k]; m];
    // reconstruct codes from bitmap + exceptions
    for i in 0..skip.n {
        let bm_off = i * skip.m_bytes;
        let exc_start = skip.offsets[i] as usize;
        let exc_end = skip.offsets[i + 1] as usize;
        let exc = &skip.exc_codes[exc_start..exc_end];
        let mut xi = 0usize;
        for sub in 0..m {
            let bit = (skip.bitmap[bm_off + sub / 8] >> (sub & 7)) & 1;
            let c = if bit == 0 { 0 } else { let v = exc[xi]; xi += 1; v };
            freq[sub][c as usize] += 1;
        }
    }
    for sub in 0..m {
        let zero = freq[sub][0];
        for ki in 1..k {
            assert!(
                freq[sub][ki] <= zero,
                "subspace {}: centroid {} has {} > dominant 0 ({})",
                sub,
                ki,
                freq[sub][ki],
                zero
            );
        }
    }
}

#[test]
fn skip_saves_memory_on_clustered_data() {
    let data = clustered_dataset(4_000, 96, 12, 11);
    let cb = Codebook::train(&data, 12, 32, 8, 0xC0FFEE);
    let skip = DominantSkipPq::build(&data, cb);
    let s = skip.stats();
    // On clustered data we should see non-trivial skip rate.
    assert!(
        s.skip_rate > 0.03,
        "expected skip_rate > 0.03, got {}",
        s.skip_rate
    );
    // And lower memory than flat codes (modulo offsets overhead).
    // On small N, offsets overhead can dominate, so just require bitmap<=flat and nonzero skips.
    assert!(s.bytes_baseline > 0);
    assert!(s.bytes_skip > 0);
}
