//! Exact baseline: brute-force scan over points whose attribute is in range.
//! Used both as a reference implementation and as the ground truth for
//! recall@k measurements.

use crate::data::{in_range, sq_l2, Dataset, Query};
use crate::graph::top_k;
use crate::{Hit, RangeAnn};

pub struct LinearPrefilter<'a> {
    ds: &'a Dataset,
}

impl<'a> LinearPrefilter<'a> {
    pub fn new(ds: &'a Dataset) -> Self {
        Self { ds }
    }
}

impl<'a> RangeAnn for LinearPrefilter<'a> {
    fn name(&self) -> &'static str {
        "linear-prefilter"
    }

    fn search(&self, query: &Query, k: usize) -> Vec<Hit> {
        let mut cands: Vec<(f32, u32)> = Vec::new();
        for p in &self.ds.points {
            if in_range(p.attr, query.lo, query.hi) {
                let d = sq_l2(&p.vec, &query.vec);
                cands.push((d, p.id));
            }
        }
        top_k(cands, k)
            .into_iter()
            .map(|(d, id)| Hit { id, dist: d })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_respects_range() {
        let ds = Dataset::random_gaussian(500, 8, 3);
        let lin = LinearPrefilter::new(&ds);
        let q = Query { vec: vec![0.0; 8], lo: 100.0, hi: 199.0 };
        let hits = lin.search(&q, 10);
        for h in &hits {
            let attr = ds.points[h.id as usize].attr;
            assert!(attr >= 100.0 && attr <= 199.0, "id {} attr {} out of range", h.id, attr);
        }
    }

    #[test]
    fn linear_returns_sorted() {
        let ds = Dataset::random_gaussian(500, 8, 4);
        let lin = LinearPrefilter::new(&ds);
        let q = Query { vec: vec![0.0; 8], lo: 0.0, hi: 499.0 };
        let hits = lin.search(&q, 10);
        for w in hits.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
    }
}
