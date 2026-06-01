//! Search-time counters.

#[derive(Default, Debug, Clone, Copy)]
pub struct SearchStats {
    /// Number of full f32 distance computations performed.
    pub full_dist: u64,
    /// Number of neighbors visited.
    pub visited: u64,
    /// Number of candidates skipped via the triangle inequality.
    pub pruned: u64,
}

impl SearchStats {
    pub fn merge(&mut self, other: &SearchStats) {
        self.full_dist += other.full_dist;
        self.visited += other.visited;
        self.pruned += other.pruned;
    }
}
