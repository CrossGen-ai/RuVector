//! Reference Chamfer similarity, used as ground truth in tests/benches.

/// Asymmetric Chamfer similarity: sum over q in `query` of max over d in `doc`
/// of inner product. This is the score ColBERT/ColPali optimize.
pub fn chamfer_similarity(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    let mut total = 0.0_f32;
    for q in query {
        let mut best = f32::NEG_INFINITY;
        for d in doc {
            let mut s = 0.0_f32;
            // length must match — caller responsibility (validated by encoder).
            for i in 0..q.len() {
                s += q[i] * d[i];
            }
            if s > best {
                best = s;
            }
        }
        total += best;
    }
    total
}

/// Plain inner product on two equal-length vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0_f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}
