//! Correctness tests: every backend must match brute-force on small data,
//! and the segment-tree must respect range boundaries exactly.

use ruvector_serf::{
    flat::Flat,
    nsw::NswParams,
    nsw_post::NswPost,
    recall,
    segment::SegmentGraph,
    Range, RangeAnn,
};
use std::sync::Arc;

const D: usize = 16;
const N: usize = 600;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
}

fn make_dataset() -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut s = 0xDEADBEEFu64;
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| (0..D).map(|_| lcg(&mut s)).collect()).collect();
    let keys: Vec<f32> = (0..N).map(|i| i as f32 / N as f32).collect();
    (vectors, keys)
}

#[test]
fn segment_graph_respects_range() {
    let (vectors, keys) = make_dataset();
    let varc = Arc::new(vectors);
    let params = NswParams::default();
    let seg = SegmentGraph::build(varc.clone(), keys.clone(), params, 64);
    let q = vec![0.0f32; D];
    let range = Range { lo: 0.3, hi: 0.6 };
    let res = seg.search(&q, range, 20);
    assert!(!res.is_empty());
    for (id, _) in &res {
        assert!(
            keys[*id] >= range.lo && keys[*id] <= range.hi,
            "id {id} key {} outside range {:?}",
            keys[*id],
            range
        );
    }
}

#[test]
fn segment_graph_recall_matches_flat_full_range() {
    let (vectors, keys) = make_dataset();
    let varc = Arc::new(vectors.clone());
    let params = NswParams {
        m: 16,
        ef_construction: 64,
        ef_search: 64,
    };
    let flat = Flat::new(vectors, keys.clone());
    let seg = SegmentGraph::build(varc, keys, params, 64);
    let mut s = 0xABCDu64;
    let mut total = 0.0f32;
    let nq = 25;
    let range = Range { lo: 0.0, hi: 1.0 };
    for _ in 0..nq {
        let q: Vec<f32> = (0..D).map(|_| lcg(&mut s)).collect();
        let truth = flat.search(&q, range, 10);
        let got = seg.search(&q, range, 10);
        total += recall(&got, &truth);
    }
    let avg = total / nq as f32;
    assert!(avg >= 0.80, "segment-graph recall@10 on full range = {avg}");
}

#[test]
fn nsw_postfilter_correctness_on_full_range() {
    let (vectors, keys) = make_dataset();
    let varc = Arc::new(vectors.clone());
    let params = NswParams::default();
    let flat = Flat::new(vectors, keys.clone());
    let nsw = NswPost::build(varc, keys, params, 4);
    let mut s = 0x9999u64;
    let q: Vec<f32> = (0..D).map(|_| lcg(&mut s)).collect();
    let range = Range { lo: 0.0, hi: 1.0 };
    let truth = flat.search(&q, range, 10);
    let got = nsw.search(&q, range, 10);
    assert!(recall(&got, &truth) >= 0.7);
}

#[test]
fn flat_top1_is_globally_nearest_within_range() {
    let (vectors, keys) = make_dataset();
    let flat = Flat::new(vectors.clone(), keys.clone());
    let q = vec![0.5f32; D];
    let range = Range { lo: 0.2, hi: 0.8 };
    let got = flat.search(&q, range, 1);
    let best = got[0].0;
    // Brute over the same range manually.
    let mut bf: Vec<(usize, f32)> = (0..N)
        .filter(|i| range.contains(keys[*i]))
        .map(|i| {
            let v = &vectors[i];
            let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum();
            (i, d)
        })
        .collect();
    bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    assert_eq!(best, bf[0].0);
}
