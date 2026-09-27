//! Split policies. A `SplitPolicy` is asked, once per probed cell per search,
//! whether that cell should be split into two child cells right now.

/// Decision returned by a policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitDecision {
    /// Do not split the cell during this search.
    Keep,
    /// Split the cell into two child cells before the next search.
    Split,
}

/// Trait implemented by cell-split policies.
pub trait SplitPolicy {
    fn decide(
        &mut self,
        cell_id: usize,
        cell_size: usize,
        probe_count: u64,
        total_cells: usize,
        total_vectors: usize,
    ) -> SplitDecision;
}

/// Baseline: never split. Equivalent to a static IVF index.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoSplit;

impl SplitPolicy for NoSplit {
    fn decide(
        &mut self,
        _cell_id: usize,
        _cell_size: usize,
        _probe_count: u64,
        _total_cells: usize,
        _total_vectors: usize,
    ) -> SplitDecision {
        SplitDecision::Keep
    }
}

/// Fixed threshold: split when `probe_count >= probe_thresh` **and**
/// `cell_size >= size_thresh`.
#[derive(Debug, Clone, Copy)]
pub struct FixedThreshold {
    probe_thresh: u64,
    size_thresh: usize,
}

impl FixedThreshold {
    pub fn new(probe_thresh: u64, size_thresh: usize) -> Self {
        Self { probe_thresh, size_thresh }
    }
}

impl SplitPolicy for FixedThreshold {
    fn decide(
        &mut self,
        _cell_id: usize,
        cell_size: usize,
        probe_count: u64,
        _total_cells: usize,
        _total_vectors: usize,
    ) -> SplitDecision {
        if probe_count >= self.probe_thresh && cell_size >= self.size_thresh {
            SplitDecision::Split
        } else {
            SplitDecision::Keep
        }
    }
}

/// Adaptive threshold: the split trigger scales with cell size relative to the
/// dataset. Larger cells split earlier; small cells are protected by
/// `min_size`. Concretely:
///
/// ```text
/// eff_probe_thresh = max(min_probe,
///                        floor(base_probe * (mean_size / cell_size)))
/// ```
///
/// Where `mean_size = total_vectors / total_cells`. Hot cells that already
/// carry a lot of members get cracked aggressively; nearly-empty cells are
/// left alone to avoid oscillation.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveThreshold {
    base_probe: u64,
    min_probe: u64,
    min_size: usize,
}

impl AdaptiveThreshold {
    pub fn new(base_probe: u64, min_probe: u64, min_size: usize) -> Self {
        Self { base_probe, min_probe, min_size }
    }
}

impl SplitPolicy for AdaptiveThreshold {
    fn decide(
        &mut self,
        _cell_id: usize,
        cell_size: usize,
        probe_count: u64,
        total_cells: usize,
        total_vectors: usize,
    ) -> SplitDecision {
        if cell_size < self.min_size || total_cells == 0 {
            return SplitDecision::Keep;
        }
        let mean_size = (total_vectors as f64 / total_cells as f64).max(1.0);
        let ratio = mean_size / (cell_size as f64).max(1.0);
        let eff = ((self.base_probe as f64) * ratio).floor() as u64;
        let eff = eff.max(self.min_probe);
        if probe_count >= eff {
            SplitDecision::Split
        } else {
            SplitDecision::Keep
        }
    }
}
