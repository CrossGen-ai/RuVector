//! Dataset and distance primitives.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, StandardNormal};

/// A vector plus a single ordinal attribute (e.g. unix-timestamp index).
#[derive(Debug, Clone)]
pub struct Point {
    pub id: u32,
    pub vec: Vec<f32>,
    /// Attribute is stored as an `f32` so the range bounds are continuous;
    /// in practice points are inserted with monotonically increasing values.
    pub attr: f32,
}

/// Range-filtered ANN query: nearest neighbors of `vec` whose attribute is in
/// the inclusive interval `[lo, hi]`.
#[derive(Debug, Clone)]
pub struct Query {
    pub vec: Vec<f32>,
    pub lo: f32,
    pub hi: f32,
}

/// Collection of points; provides accessors and synthetic generators.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub dim: usize,
    pub points: Vec<Point>,
}

impl Dataset {
    pub fn new(dim: usize) -> Self {
        Self { dim, points: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Generate `n` gaussian-random vectors of dimension `dim`. Attribute is
    /// the insertion index, mimicking a monotonically-growing timestamp.
    pub fn random_gaussian(n: usize, dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut ds = Dataset::new(dim);
        ds.points.reserve(n);
        for i in 0..n {
            let mut v = Vec::with_capacity(dim);
            for _ in 0..dim {
                let x: f32 = StandardNormal.sample(&mut rng);
                v.push(x);
            }
            ds.points.push(Point { id: i as u32, vec: v, attr: i as f32 });
        }
        ds
    }

    /// Generate `q` random queries with random closed attribute ranges.
    /// `range_frac` controls the typical span of the range as a fraction of
    /// `[0, n)`. Lower values produce narrower (harder) range filters.
    pub fn random_queries(
        &self,
        q: usize,
        range_frac: f32,
        seed: u64,
    ) -> Vec<Query> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = self.len() as f32;
        let span = (n * range_frac).max(1.0);
        let mut queries = Vec::with_capacity(q);
        for _ in 0..q {
            let mut v = Vec::with_capacity(self.dim);
            for _ in 0..self.dim {
                let x: f32 = StandardNormal.sample(&mut rng);
                v.push(x);
            }
            let lo: f32 = rng.gen_range(0.0..(n - span).max(1.0));
            let hi: f32 = (lo + span).min(n - 1.0);
            queries.push(Query { vec: v, lo, hi });
        }
        queries
    }
}

/// Squared L2 distance. Unrolled for the hot path.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    let mut i = 0;
    while i + 4 <= a.len() {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < a.len() {
        let d = a[i] - b[i];
        sum += d * d;
        i += 1;
    }
    sum
}

#[inline]
pub fn in_range(attr: f32, lo: f32, hi: f32) -> bool {
    attr >= lo && attr <= hi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_zero_on_self() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(sq_l2(&v, &v) < 1e-6);
    }

    #[test]
    fn dist_pythagoras() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((sq_l2(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn dataset_random_shapes() {
        let ds = Dataset::random_gaussian(100, 8, 42);
        assert_eq!(ds.len(), 100);
        assert_eq!(ds.points[0].vec.len(), 8);
        assert_eq!(ds.points[99].attr, 99.0);
    }

    #[test]
    fn queries_have_valid_ranges() {
        let ds = Dataset::random_gaussian(1_000, 4, 1);
        let qs = ds.random_queries(20, 0.1, 7);
        assert_eq!(qs.len(), 20);
        for q in &qs {
            assert!(q.lo <= q.hi);
            assert!(q.lo >= 0.0);
            assert!(q.hi < 1000.0);
        }
    }

    #[test]
    fn in_range_inclusive() {
        assert!(in_range(5.0, 0.0, 10.0));
        assert!(in_range(0.0, 0.0, 10.0));
        assert!(in_range(10.0, 0.0, 10.0));
        assert!(!in_range(11.0, 0.0, 10.0));
    }
}
