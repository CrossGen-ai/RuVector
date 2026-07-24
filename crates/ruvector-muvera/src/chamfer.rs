//! Exact Chamfer / MaxSim similarity and helper norms.
//!
//! Given a query set `Q ⊆ R^d` and a document set `D ⊆ R^d`, the
//! Chamfer / MaxSim similarity used by ColBERT-style late interaction is
//!
//! ```text
//! MaxSim(Q, D) = Σ_{q ∈ Q}  max_{v ∈ D}  ⟨q, v⟩
//! ```
//!
//! When each vector is L2-normalized this is equal to
//! `Σ_q max_v cos(q, v)`, i.e. cosine MaxSim, which is what ColBERTv2
//! actually deploys.

/// Exact MaxSim / Chamfer inner-product similarity.
///
/// `query` and `doc` are laid out as row-major `[n_Q × d]` and `[n_D × d]`
/// flat `f32` slices. Returns `Σ_q max_v ⟨q, v⟩`.
///
/// # Panics
/// If `query.len() % d != 0` or `doc.len() % d != 0`.
pub fn maxsim(query: &[f32], doc: &[f32], d: usize) -> f32 {
    assert!(d > 0, "dimensionality must be positive");
    assert!(query.len() % d == 0, "query.len() must be multiple of d");
    assert!(doc.len() % d == 0, "doc.len() must be multiple of d");
    let n_q = query.len() / d;
    let n_d = doc.len() / d;
    if n_q == 0 || n_d == 0 {
        return 0.0;
    }
    let mut total = 0.0f32;
    for qi in 0..n_q {
        let q = &query[qi * d..(qi + 1) * d];
        let mut best = f32::NEG_INFINITY;
        for di in 0..n_d {
            let v = &doc[di * d..(di + 1) * d];
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += q[j] * v[j];
            }
            if acc > best {
                best = acc;
            }
        }
        total += best;
    }
    total
}

/// L2-normalize a single vector in place. Vectors with zero norm are left
/// untouched (they map to themselves).
pub fn l2_normalize_inplace(v: &mut [f32]) {
    let mut sq = 0.0f32;
    for x in v.iter() {
        sq += x * x;
    }
    if sq > 0.0 {
        let inv = 1.0 / sq.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// L2-normalize every vector in a `[n × d]` flat set.
pub fn l2_normalize_set(set: &mut [f32], d: usize) {
    for chunk in set.chunks_exact_mut(d) {
        l2_normalize_inplace(chunk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxsim_identity_pair() {
        // If Q == D and vectors are orthonormal basis vectors, MaxSim = |Q|
        let d = 4;
        let q = vec![
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
        ];
        let doc = q.clone();
        let s = maxsim(&q, &doc, d);
        assert!((s - 2.0).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn maxsim_prefers_best_doc_token() {
        let d = 2;
        // Query: single token (1, 0).
        let q = vec![1.0, 0.0];
        // Doc: (0.5, 0.5) and (0.9, 0.1). max IP = 0.9.
        let doc = vec![0.5, 0.5, 0.9, 0.1];
        let s = maxsim(&q, &doc, d);
        assert!((s - 0.9).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn maxsim_empty_sides_zero() {
        let d = 3;
        assert_eq!(maxsim(&[], &[1.0, 2.0, 3.0], d), 0.0);
        assert_eq!(maxsim(&[1.0, 2.0, 3.0], &[], d), 0.0);
    }

    #[test]
    fn l2_normalize_zero_vector_untouched() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize_inplace(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_scales_to_unit() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize_inplace(&mut v);
        let n: f32 = v.iter().map(|x| x * x).sum();
        assert!((n - 1.0).abs() < 1e-6);
    }
}
