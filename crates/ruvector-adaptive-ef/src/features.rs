//! Cheap online features used by the adaptive ef predictors.
//!
//! Computing these features is *fast* (a handful of dot products against
//! a small set of pre-selected pivot vectors), so the predictor overhead
//! is dominated by the savings from skipping unnecessary graph hops.

use crate::hnsw::squared_l2;

/// Online query difficulty features.
///
/// All features are intentionally cheap (O(P · D) where P ≈ 8 pivots and
/// D is the embedding dim).  They are designed to be informative of how
/// many graph hops the search will need before recall saturates.
#[derive(Debug, Clone, Copy)]
pub struct QueryFeatures {
    /// Minimum squared L2 distance to any pivot.  Proxy for "is this query
    /// close to known data?"  Larger ⇒ likely OOD ⇒ harder.
    pub min_pivot_d2: f32,
    /// Mean squared L2 distance to all pivots.  Picks up global density.
    pub mean_pivot_d2: f32,
    /// Standard deviation of pivot distances.  Picks up anisotropy /
    /// cluster boundary cases.
    pub std_pivot_d2: f32,
    /// Ratio min / mean.  Scale-invariant difficulty signal — small ratio
    /// (close to 1.0) means "query is equidistant from all clusters" ⇒
    /// boundary case, harder.
    pub min_over_mean: f32,
}

impl QueryFeatures {
    /// Pack features into the 5-dimensional input vector used by the
    /// linear predictor (`[1, min_d2, mean_d2, std_d2, min_over_mean]`).
    /// The leading `1` is the bias term.
    #[inline]
    pub fn to_input(&self) -> [f32; 5] {
        [
            1.0,
            self.min_pivot_d2,
            self.mean_pivot_d2,
            self.std_pivot_d2,
            self.min_over_mean,
        ]
    }
}

/// Compute online features for `query` against the given pivot set.
///
/// # Panics
/// Panics if `pivots` is empty or any pivot has a different length than
/// `query`.
pub fn extract_features(query: &[f32], pivots: &[Vec<f32>]) -> QueryFeatures {
    assert!(!pivots.is_empty(), "must provide at least one pivot");
    let dim = query.len();

    let mut min_d2 = f32::INFINITY;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for p in pivots {
        assert_eq!(p.len(), dim, "pivot dim mismatch");
        let d2 = squared_l2(query, p);
        if d2 < min_d2 {
            min_d2 = d2;
        }
        sum += d2;
        sum_sq += d2 * d2;
    }
    let n = pivots.len() as f32;
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    let std = var.sqrt();
    let min_over_mean = if mean > 0.0 { min_d2 / mean } else { 0.0 };

    QueryFeatures {
        min_pivot_d2: min_d2,
        mean_pivot_d2: mean,
        std_pivot_d2: std,
        min_over_mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_are_finite_and_consistent() {
        let pivots = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let q = vec![0.5, 0.5, 0.5];
        let f = extract_features(&q, &pivots);
        assert!(f.min_pivot_d2.is_finite());
        assert!(f.mean_pivot_d2.is_finite());
        assert!(f.std_pivot_d2.is_finite());
        assert!(f.min_over_mean >= 0.0 && f.min_over_mean <= 1.0);
        assert!(f.min_pivot_d2 <= f.mean_pivot_d2);
    }

    #[test]
    fn boundary_query_has_high_min_over_mean() {
        // Query equidistant from all pivots → boundary case.
        let pivots = vec![
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, -1.0],
        ];
        let q = vec![0.0, 0.0];
        let f = extract_features(&q, &pivots);
        // All distances equal ⇒ min == mean ⇒ ratio == 1.
        assert!((f.min_over_mean - 1.0).abs() < 1e-5);
    }
}
