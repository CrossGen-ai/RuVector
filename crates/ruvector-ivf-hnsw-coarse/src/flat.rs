//! Flat coarse quantizer: brute-force scan of every centroid. This is the
//! oracle we compare `HnswCoarse` and `RandomProjCoarse` against.

use crate::{sq_l2, CoarseError, IvfCoarse};

/// Flat / brute-force coarse quantizer. `top_lists` cost is `O(C · d)`.
#[derive(Debug, Default)]
pub struct FlatCoarse {
    centroids: Vec<f32>,
    d: usize,
    c: usize,
}

impl FlatCoarse {
    /// Construct an empty [`FlatCoarse`]. Call [`IvfCoarse::fit`] before use.
    pub fn new() -> Self {
        Self::default()
    }
}

impl IvfCoarse for FlatCoarse {
    fn fit(&mut self, centroids: &[f32], d: usize) -> Result<(), CoarseError> {
        if d == 0 || centroids.len() % d != 0 {
            return Err(CoarseError::DimensionMismatch {
                expected: d,
                got: centroids.len(),
            });
        }
        self.centroids = centroids.to_vec();
        self.d = d;
        self.c = centroids.len() / d;
        Ok(())
    }

    fn top_lists(&self, query: &[f32], nprobe: usize) -> Vec<u32> {
        assert_eq!(query.len(), self.d);
        let nprobe = nprobe.min(self.c);
        let mut scored: Vec<(f32, u32)> = (0..self.c)
            .map(|i| {
                let off = i * self.d;
                (sq_l2(query, &self.centroids[off..off + self.d]), i as u32)
            })
            .collect();
        // Partial sort: nth_element then sort head.
        let pivot = nprobe.saturating_sub(1).min(scored.len() - 1);
        scored.select_nth_unstable_by(pivot, |a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut head: Vec<(f32, u32)> = scored.into_iter().take(nprobe).collect();
        head.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        head.into_iter().map(|(_, i)| i).collect()
    }

    fn name(&self) -> &'static str {
        "FlatCoarse"
    }

    fn extra_bytes(&self) -> usize {
        0
    }
}
