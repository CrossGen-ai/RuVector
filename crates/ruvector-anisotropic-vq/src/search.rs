//! Top-k search helpers + recall metric used by the benchmark binary.

use crate::pq::ProductQuantizer;

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
}

/// Exact brute-force inner-product top-k.
pub fn brute_force_topk(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<Hit> {
    let mut scored: Vec<Hit> = data
        .iter()
        .enumerate()
        .map(|(i, v)| Hit {
            id: i as u32,
            score: ip(query, v),
        })
        .collect();
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    scored.truncate(k);
    scored
}

/// PQ-based top-k using a precomputed inner-product LUT.
pub fn pq_topk(query: &[f32], codes: &[Vec<u8>], pq: &ProductQuantizer, k: usize) -> Vec<Hit> {
    let lut = pq.build_ip_lut(query);
    let mut scored: Vec<Hit> = codes
        .iter()
        .enumerate()
        .map(|(i, c)| Hit {
            id: i as u32,
            score: pq.score_code(&lut, c),
        })
        .collect();
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    scored.truncate(k);
    scored
}

/// Recall @ k of `approx` against ground truth `gt`.
pub fn recall_at_k(gt: &[Hit], approx: &[Hit], k: usize) -> f32 {
    let g: std::collections::HashSet<u32> = gt.iter().take(k).map(|h| h.id).collect();
    let hit = approx.iter().take(k).filter(|h| g.contains(&h.id)).count();
    hit as f32 / k as f32
}

fn ip(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
