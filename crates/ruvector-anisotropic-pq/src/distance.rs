//! Asymmetric Distance Computation (ADC) helpers.
//!
//! Given a quantizer `Q` and a query `q`, we precompute one `m x k` table of
//! sub-inner-products `T[s][c] = <q_s, codebook[s][c]>`. The approximate dot
//! product `<q, x>` for any encoded vector reduces to `m` table lookups and
//! `m - 1` additions — far cheaper than reconstructing `x` and computing the
//! full inner product.

use crate::Quantizer;

/// Rank `top_k` codes by approximate inner product, descending.
pub fn topk_by_dot<Q: Quantizer>(
    q: &Q,
    query: &[f32],
    codes: &[Vec<u8>],
    top_k: usize,
) -> Vec<u32> {
    let table = q.dot_table(query);
    let mut scored: Vec<(u32, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, code)| (i as u32, q.dot_with_table(&table, code)))
        .collect();
    // Partial sort: full sort is fine for PoC dataset sizes; production
    // would use a bounded max-heap. We sort descending by score.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(top_k).map(|(id, _)| id).collect()
}

/// Brute-force ground truth top-k by exact inner product.
pub fn topk_exact_dot(database: &[Vec<f32>], query: &[f32], top_k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let mut s = 0.0f32;
            for j in 0..v.len() { s += v[j] * query[j]; }
            (i as u32, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(top_k).map(|(id, _)| id).collect()
}
