//! Baseline estimator: exact inner-product per neighbour.
//!
//! Provided for apples-to-apples recall comparisons and as the "gold" against
//! which the FINGER/JL variants are measured. Retains the original vectors so
//! bytes_per_vector = 4 * d.

use crate::estimator::{DistanceEstimator, PivotHandle};
use crate::{dot, PivotIndex};

pub struct ExactEstimator<'a> {
    idx: &'a PivotIndex,
}

impl<'a> ExactEstimator<'a> {
    pub fn new(idx: &'a PivotIndex) -> Self { Self { idx } }
}

struct ExactHandle<'a> {
    idx: &'a PivotIndex,
    q: Vec<f32>,
}

impl<'a> PivotHandle for ExactHandle<'a> {
    fn score(&self, neighbour: u32) -> f32 {
        dot(&self.q, &self.idx.vectors[neighbour as usize])
    }
}

impl<'a> DistanceEstimator for ExactEstimator<'a> {
    fn name(&self) -> &'static str { "exact" }
    fn bytes_per_vector(&self) -> usize { self.idx.dim * std::mem::size_of::<f32>() }
    fn prepare_query<'b>(&'b self, query: &[f32], _pivot_id: u32) -> Box<dyn PivotHandle + 'b> {
        Box::new(ExactHandle { idx: self.idx, q: query.to_vec() })
    }
}
