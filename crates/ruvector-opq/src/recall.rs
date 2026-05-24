//! ANN evaluation helpers: brute-force ground truth + recall@k via ADC scan.

use crate::Quantizer;

/// Build `n × k` ground-truth neighbour ids using exhaustive squared-L2 search
/// against `base` (n_base × d). Returns row-major `n_query × k`.
pub fn ground_truth(
    base: &[f32],
    queries: &[f32],
    n_base: usize,
    n_query: usize,
    d: usize,
    k: usize,
) -> Vec<u32> {
    let mut gt = vec![0u32; n_query * k];
    let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(n_base);
    for q in 0..n_query {
        let qv = &queries[q * d..(q + 1) * d];
        scratch.clear();
        for i in 0..n_base {
            let bv = &base[i * d..(i + 1) * d];
            let mut s = 0.0f32;
            for j in 0..d {
                let e = qv[j] - bv[j];
                s += e * e;
            }
            scratch.push((s, i as u32));
        }
        scratch.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        for j in 0..k {
            gt[q * k + j] = scratch[j].1;
        }
    }
    gt
}

/// Recall@k of a quantizer using ADC over all base codes.
/// `codes` is `n_base × m` flat.
pub fn recall_at_k<Q: Quantizer>(
    q: &Q,
    codes: &[u8],
    n_base: usize,
    queries: &[f32],
    n_query: usize,
    d: usize,
    gt: &[u32],
    k: usize,
) -> f32 {
    let m = q.m();
    let mut hits = 0usize;
    let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(n_base);
    let mut lut = vec![0.0f32; m * 256];
    for qi in 0..n_query {
        let qv = &queries[qi * d..(qi + 1) * d];
        q.build_lut(qv, &mut lut);
        scratch.clear();
        for i in 0..n_base {
            let code = &codes[i * m..(i + 1) * m];
            scratch.push((q.adc_lut(&lut, code), i as u32));
        }
        scratch.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let gt_row = &gt[qi * k..(qi + 1) * k];
        for j in 0..k {
            if gt_row.contains(&scratch[j].1) {
                hits += 1;
            }
        }
    }
    hits as f32 / (n_query * k) as f32
}
