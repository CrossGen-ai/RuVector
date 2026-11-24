//! Linear-scan baseline: exact range search in O(n*d).

use crate::{sqr_l2, RangeHit, RangeIndex};

pub struct LinearScan {
    points: Vec<Vec<f32>>,
}

impl RangeIndex for LinearScan {
    fn build(points: &[Vec<f32>]) -> Self {
        Self {
            points: points.to_vec(),
        }
    }

    fn range_search(&self, query: &[f32], r2: f32) -> Vec<RangeHit> {
        let mut out = Vec::new();
        for (i, p) in self.points.iter().enumerate() {
            let d = sqr_l2(query, p);
            if d <= r2 {
                out.push((i as u32, d));
            }
        }
        out
    }

    fn name(&self) -> &'static str {
        "linear-scan"
    }
}
