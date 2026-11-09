use crate::quant::Quantizer;

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() { let d = a[i] - b[i]; s += d * d; }
    s
}

pub fn topk_exact(queries: &[Vec<f32>], base: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(queries.len());
    for q in queries {
        let mut heap: Vec<(f32, usize)> = base.iter().enumerate()
            .map(|(i, b)| (l2_sq(q, b), i)).collect();
        heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out.push(heap.iter().take(k).map(|&(_, i)| i).collect());
    }
    out
}

pub fn topk_via_decoded<Q: Quantizer>(q: &Q, queries: &[Vec<f32>], base_codes: &[Vec<u8>], k: usize) -> Vec<Vec<usize>> {
    let decoded: Vec<Vec<f32>> = base_codes.iter().map(|c| q.decode(c)).collect();
    topk_exact(queries, &decoded, k)
}

#[derive(Debug, Clone, Copy)]
pub struct EvalReport {
    pub recall_at_k: f64,
    pub avg_mse_per_dim: f64,
}

pub fn recall_at_k(gold: &[Vec<usize>], pred: &[Vec<usize>], k: usize) -> f64 {
    let mut hits = 0usize;
    let mut total = 0usize;
    for (g, p) in gold.iter().zip(pred.iter()) {
        let gset: std::collections::HashSet<_> = g.iter().take(k).copied().collect();
        for &pi in p.iter().take(k) {
            total += 1;
            if gset.contains(&pi) { hits += 1; }
        }
    }
    (hits as f64) / (total as f64)
}

pub fn evaluate<Q: Quantizer>(q: &Q, base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> EvalReport {
    let codes: Vec<Vec<u8>> = base.iter().map(|v| q.encode(v)).collect();
    let decoded: Vec<Vec<f32>> = codes.iter().map(|c| q.decode(c)).collect();
    let gold = topk_exact(queries, base, k);
    let pred = topk_exact(queries, &decoded, k);
    let rec = recall_at_k(&gold, &pred, k);
    let mut mse_sum = 0.0f64;
    let mut dims = 0usize;
    for (v, r) in base.iter().zip(decoded.iter()) {
        for i in 0..v.len() {
            let d = (v[i] - r[i]) as f64;
            mse_sum += d * d;
            dims += 1;
        }
    }
    EvalReport { recall_at_k: rec, avg_mse_per_dim: mse_sum / dims as f64 }
}
