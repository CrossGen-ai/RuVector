//! Brute-force baseline: scan everything, postfilter by range.

use crate::{sq_l2, Range, RangeAnn};

pub struct Flat {
    pub vectors: Vec<Vec<f32>>,
    pub keys: Vec<f32>,
}

impl Flat {
    pub fn new(vectors: Vec<Vec<f32>>, keys: Vec<f32>) -> Self {
        assert_eq!(vectors.len(), keys.len());
        Self { vectors, keys }
    }
}

impl RangeAnn for Flat {
    fn name(&self) -> &'static str {
        "flat-postfilter"
    }

    fn search(&self, q: &[f32], range: Range, k: usize) -> Vec<(usize, f32)> {
        let mut hits: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .filter(|(i, _)| range.contains(self.keys[*i]))
            .map(|(i, v)| (i, sq_l2(q, v)))
            .collect();
        hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        hits.truncate(k);
        hits
    }
}
