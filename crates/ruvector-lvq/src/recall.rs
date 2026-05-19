//! Brute-force scoring helpers + recall measurement.

use crate::distance::{l2_sq, Metric};
use crate::quantizer::{Encoded, Quantizer};

pub fn topk_exact(query: &[f32], db: &[Vec<f32>], k: usize, metric: Metric) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = (0..db.len())
        .map(|i| {
            let s = match metric {
                Metric::L2 => l2_sq(query, &db[i]),
                Metric::Ip => -crate::distance::ip(query, &db[i]),
            };
            (i, s)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

pub fn topk_quantized<Q: Quantizer>(
    q: &Q,
    query: &[f32],
    db: &[Encoded],
    k: usize,
    metric: Metric,
) -> Vec<usize> {
    let qn = crate::distance::ip(query, query);
    let mut scored: Vec<(usize, f32)> = db
        .iter()
        .enumerate()
        .map(|(i, e)| (i, q.distance(query, qn, e, metric)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

pub fn recall_at(truth: &[usize], approx: &[usize], k: usize) -> f32 {
    let kk = k.min(truth.len()).min(approx.len());
    let t: std::collections::HashSet<usize> = truth.iter().take(kk).copied().collect();
    let hit = approx.iter().take(kk).filter(|i| t.contains(i)).count();
    hit as f32 / kk as f32
}
