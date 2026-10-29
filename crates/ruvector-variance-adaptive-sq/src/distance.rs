//! L2 distance helpers and ground-truth / recall utilities.

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

pub fn brute_force_topk(
    queries: &[f32],
    base: &[f32],
    dim: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    let nq = queries.len() / dim;
    let nb = base.len() / dim;
    let mut out = Vec::with_capacity(nq);
    for q in 0..nq {
        let mut heap: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        let qv = &queries[q * dim..(q + 1) * dim];
        for b in 0..nb {
            let bv = &base[b * dim..(b + 1) * dim];
            let d = l2_sq(qv, bv);
            if heap.len() < k {
                heap.push((d, b as u32));
                heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            } else if d < heap[k - 1].0 {
                heap[k - 1] = (d, b as u32);
                heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        out.push(heap.into_iter().map(|(_, i)| i).collect());
    }
    out
}

pub fn recall_at_k(gt: &[Vec<u32>], pred: &[Vec<u32>], k: usize) -> f32 {
    let n = gt.len();
    let mut hits = 0usize;
    let mut total = 0usize;
    for i in 0..n {
        let g: std::collections::HashSet<u32> = gt[i].iter().take(k).copied().collect();
        for p in pred[i].iter().take(k) {
            if g.contains(p) {
                hits += 1;
            }
        }
        total += k;
    }
    hits as f32 / total as f32
}
