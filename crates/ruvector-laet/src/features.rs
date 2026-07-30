//! Per-iteration search-traversal features fed to the learned stopping predictor.
//!
//! Deliberately cheap to compute: 5 scalars per iteration, no extra distance calls.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Features {
    /// Iteration index (candidates popped so far).
    pub iter: f32,
    /// Best (min) distance in the result heap right now.
    pub best_dist: f32,
    /// best_dist decrease since previous iteration (>=0). Larger => still improving.
    pub delta_best_dist: f32,
    /// Iterations elapsed since best_dist last strictly improved.
    pub iters_since_improve: f32,
    /// Ratio of iteration to a nominal ef_min (early-stop calibration constant).
    pub ef_min_reached: f32,
}

impl Features {
    pub const DIM: usize = 5;

    #[inline]
    pub fn as_array(&self) -> [f32; Self::DIM] {
        [
            self.iter,
            self.best_dist,
            self.delta_best_dist,
            self.iters_since_improve,
            self.ef_min_reached,
        ]
    }
}
