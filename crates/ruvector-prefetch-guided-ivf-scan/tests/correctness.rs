//! Integration tests: all three scan variants must return identical top-k
//! across a range of configurations. Any divergence would mean the prefetch
//! path is reading through the wrong pointer.

use ruvector_prefetch_guided_ivf_scan::{
    scan_adaptive_prefetch, scan_fixed_prefetch, scan_no_prefetch, PostingList,
};

struct Xor64(u64);
impl Xor64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 / (1u32 << 23) as f32) - 1.0
    }
}

fn make_list(dim: usize, n: usize, seed: u64) -> PostingList {
    let mut rng = Xor64::new(seed);
    let mut v = Vec::with_capacity(dim * n);
    for _ in 0..(dim * n) {
        v.push(rng.next_f32());
    }
    PostingList::new(dim, v)
}

fn make_query(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xor64::new(seed);
    (0..dim).map(|_| rng.next_f32()).collect()
}

#[test]
fn variants_agree_across_dims_and_sizes() {
    for &dim in &[16usize, 64, 128, 256, 512] {
        for &n in &[1usize, 2, 7, 33, 1024, 4096] {
            let list = make_list(dim, n, 0xDEAD_BEEF ^ dim as u64 ^ n as u64);
            let q = make_query(dim, 0xCAFE_F00D ^ dim as u64 ^ n as u64);
            for &k in &[1usize, 5, 10] {
                let base = scan_no_prefetch(&list, &q, k);
                let f1 = scan_fixed_prefetch(&list, &q, k, 1);
                let f8 = scan_fixed_prefetch(&list, &q, k, 8);
                let f16 = scan_fixed_prefetch(&list, &q, k, 16);
                let adap = scan_adaptive_prefetch(&list, &q, k);
                assert_eq!(base, f1, "dim={dim} n={n} k={k} fixed(1) mismatch");
                assert_eq!(base, f8, "dim={dim} n={n} k={k} fixed(8) mismatch");
                assert_eq!(base, f16, "dim={dim} n={n} k={k} fixed(16) mismatch");
                assert_eq!(base, adap, "dim={dim} n={n} k={k} adaptive mismatch");
            }
        }
    }
}

#[test]
fn recall_is_exact_for_flat_scan() {
    // Flat scan is exact by construction. If any variant returned a hit not
    // in the true nearest set we'd catch it here.
    let dim = 32;
    let n = 5_000;
    let list = make_list(dim, n, 42);
    let q = make_query(dim, 7);

    let k = 20;
    let hits = scan_adaptive_prefetch(&list, &q, k);

    // Brute-force ground truth via baseline (also exact).
    let truth = scan_no_prefetch(&list, &q, k);

    let hits_ids: Vec<u32> = hits.iter().map(|h| h.idx).collect();
    let truth_ids: Vec<u32> = truth.iter().map(|h| h.idx).collect();
    assert_eq!(hits_ids, truth_ids);
}

#[test]
fn prefetch_beyond_end_is_safe() {
    // Lookahead much larger than the list. Should not crash and should return
    // the same top-k as baseline.
    let list = make_list(128, 10, 0);
    let q = make_query(128, 1);
    let baseline = scan_no_prefetch(&list, &q, 5);
    let with_huge_la = scan_fixed_prefetch(&list, &q, 5, 32);
    assert_eq!(baseline, with_huge_la);
}
