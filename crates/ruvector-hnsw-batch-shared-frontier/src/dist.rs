//! Squared-L2 distance kernels — scalar and batched.
//!
//! The batched kernel is the crux of this crate's optimization: given one
//! candidate vector `v` and `Q` query vectors, we compute all `Q` distances
//! in a single pass that touches `v` from memory once and keeps queries hot
//! in L1. That amortizes the per-node vector load cost by a factor of ~`Q`
//! on the memory subsystem (though the FMA count is unchanged).

/// Scalar squared-L2 between two `dim`-vectors.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    // 8-wide manual unroll — good enough to give the autovectorizer a hint.
    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let d0 = a[i    ] - b[i    ];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        let d4 = a[i + 4] - b[i + 4];
        let d5 = a[i + 5] - b[i + 5];
        let d6 = a[i + 6] - b[i + 6];
        let d7 = a[i + 7] - b[i + 7];
        acc += d0*d0 + d1*d1 + d2*d2 + d3*d3
             + d4*d4 + d5*d5 + d6*d6 + d7*d7;
        i += 8;
    }
    while i < n {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    acc
}

/// Batched kernel: for every query in `queries`, write the squared-L2 to
/// `candidate` into the matching slot of `out`.
///
/// This is the memory-amortizing kernel: `candidate` is streamed once per
/// call, then reused across every query.
pub fn l2_sq_batch(candidate: &[f32], queries: &[Vec<f32>], out: &mut [f32]) {
    debug_assert_eq!(queries.len(), out.len());
    // Process 4 query rows at once to keep candidate elements hot in
    // registers across queries. This is a straightforward reduction of
    // "for q { for i { ... } }" that pins the candidate load between
    // multiple query reductions.
    let q_len = queries.len();
    let dim = candidate.len();
    let mut qi = 0;
    while qi + 4 <= q_len {
        let (q0, q1, q2, q3) = (
            queries[qi].as_slice(),
            queries[qi + 1].as_slice(),
            queries[qi + 2].as_slice(),
            queries[qi + 3].as_slice(),
        );
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut i = 0;
        while i < dim {
            let c = candidate[i];
            let d0 = c - q0[i];
            let d1 = c - q1[i];
            let d2 = c - q2[i];
            let d3 = c - q3[i];
            a0 += d0 * d0;
            a1 += d1 * d1;
            a2 += d2 * d2;
            a3 += d3 * d3;
            i += 1;
        }
        out[qi    ] = a0;
        out[qi + 1] = a1;
        out[qi + 2] = a2;
        out[qi + 3] = a3;
        qi += 4;
    }
    while qi < q_len {
        out[qi] = l2_sq(candidate, &queries[qi]);
        qi += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_matches_naive() {
        let a: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..37).map(|i| (i as f32 - 3.0) * 0.05).collect();
        let mut expect = 0.0f32;
        for i in 0..37 { let d = a[i] - b[i]; expect += d * d; }
        let got = l2_sq(&a, &b);
        assert!((got - expect).abs() < 1e-4, "got {got} expect {expect}");
    }

    #[test]
    fn batch_matches_scalar() {
        let cand: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let queries: Vec<Vec<f32>> = (0..7)
            .map(|q| (0..64).map(|i| ((i + q) as f32).cos()).collect())
            .collect();
        let mut out = vec![0.0f32; queries.len()];
        l2_sq_batch(&cand, &queries, &mut out);
        for (q, o) in queries.iter().zip(out.iter()) {
            let s = l2_sq(&cand, q);
            assert!((s - o).abs() < 1e-3, "batch {o} vs scalar {s}");
        }
    }
}
