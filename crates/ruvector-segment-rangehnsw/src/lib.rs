//! ruvector-segment-rangehnsw
//!
//! Range-filtered approximate nearest neighbor search using a segment tree of
//! small proximity graphs (iRangeGraph-style, SIGMOD 2024). The core insight:
//! when filtering top-k by both vector similarity AND a continuous attribute
//! range (timestamp, price, score, etc.), neither classical HNSW post-filter
//! nor pre-filter is recall-stable across selectivities. By indexing each
//! O(log n) covering window once at build time and merging O(log n) mini-graph
//! searches at query time, we keep recall near full-HNSW while dramatically
//! reducing distance computations for narrow ranges.
//!
//! This crate ships a working PoC + benchmark binary. See
//! `examples/bench.rs` for the runnable comparison harness.

pub mod dist;
pub mod graph;
pub mod segment;

pub use graph::Graph;
pub use segment::SegmentRangeIndex;

/// Convenience: brute-force range-filtered linear scan. O(n) per query; used
/// as a recall ground truth and as a baseline variant.
pub fn brute_force_range(
    points: &[(u32, f32, Vec<f32>)],
    q: &[f32],
    lo: f32,
    hi: f32,
    k: usize,
) -> Vec<(f32, u32, f32)> {
    let mut hits: Vec<(f32, u32, f32)> = points
        .iter()
        .filter(|(_, key, _)| (lo..=hi).contains(key))
        .map(|(g, key, v)| (dist::l2_sq(q, v), *g, *key))
        .collect();
    hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    hits.truncate(k);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dist::Lcg;

    fn make_points(n: usize, d: usize, seed: u64) -> Vec<(u32, f32, Vec<f32>)> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..d).map(|_| rng.next_gauss()).collect();
                // Range key: float in [0, 1).
                let key = rng.next_f32();
                (i as u32, key, v)
            })
            .collect()
    }

    #[test]
    fn segment_index_returns_only_in_range_results() {
        let pts = make_points(800, 16, 7);
        let idx = SegmentRangeIndex::build(pts.clone(), 64, 12, 32);
        let mut rng = Lcg::new(99);
        let q: Vec<f32> = (0..16).map(|_| rng.next_gauss()).collect();
        let (lo, hi) = (0.2f32, 0.6f32);
        let res = idx.search(&q, 10, lo, hi, 32);
        assert!(!res.is_empty(), "segment index returned no results in a wide range");
        for (_d, _gid, key) in &res {
            assert!(*key >= lo && *key <= hi, "out-of-range key leaked: {}", key);
        }
    }

    #[test]
    fn segment_index_recall_against_brute_force() {
        let n = 800;
        let d = 16;
        let pts = make_points(n, d, 11);
        let idx = SegmentRangeIndex::build(pts.clone(), 64, 16, 64);
        let mut rng = Lcg::new(123);
        let mut recall_sum = 0.0f32;
        let queries = 30;
        let k = 10;
        for _ in 0..queries {
            let q: Vec<f32> = (0..d).map(|_| rng.next_gauss()).collect();
            let a = rng.next_f32();
            let b = rng.next_f32();
            let (lo, hi) = (a.min(b), a.max(b));
            let gt: Vec<u32> = brute_force_range(&pts, &q, lo, hi, k).into_iter().map(|x| x.1).collect();
            if gt.is_empty() { continue; }
            let got: Vec<u32> = idx.search(&q, k, lo, hi, 64).into_iter().map(|x| x.1).collect();
            let hits = gt.iter().filter(|g| got.contains(g)).count() as f32;
            recall_sum += hits / gt.len() as f32;
        }
        let recall = recall_sum / queries as f32;
        assert!(recall > 0.80, "PoC recall must clear 80% (was {:.3})", recall);
    }
}
