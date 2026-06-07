//! Rebalancing policies for DEDRIFT.
//!
//! All policies operate *in place* on an existing [`crate::Ivf`] and never
//! re-allocate the raw vector slab. They differ in *what* they update:
//!
//!   * [`Policy::Split`]   — split any list whose population exceeds the
//!     configured ceiling using 2-means on its members.
//!   * [`Policy::Lazy`]    — recompute the centroid of any list whose mean has
//!     drifted "far" from the stored centroid (member-weighted recenter).
//!   * [`Policy::Hybrid`]  — Lazy + Split, applied in that order.
//!
//! The DEDRIFT paper additionally performs a global k-means reassign after
//! splits; we mirror that with a [`Policy::FullRebuild`] baseline so callers
//! can measure the cost gap directly.

use crate::{nearest_centroid, sq_l2, Ivf, SmallRng};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    None,
    Split,
    Lazy,
    Hybrid,
    FullRebuild,
}

#[derive(Clone, Copy, Debug)]
pub struct PolicyConfig {
    /// Split a list whose population exceeds `split_threshold * mean_load`.
    pub split_threshold: f32,
    /// Recenter a list whose drift score is above `lazy_threshold * sigma`
    /// where sigma is the median per-list drift contribution.
    pub lazy_threshold: f32,
    /// k-means iterations used inside a 2-means split.
    pub split_kmeans_iters: usize,
    /// k-means iterations used inside a full rebuild.
    pub full_rebuild_iters: usize,
    /// Random seed for any tie-breaking.
    pub seed: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            split_threshold: 2.5,
            lazy_threshold: 1.5,
            split_kmeans_iters: 6,
            full_rebuild_iters: 8,
            seed: 0xD3D71F7,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct PolicyReport {
    pub policy: &'static str,
    pub centroids_after: usize,
    pub splits_applied: usize,
    pub lazy_recenters_applied: usize,
    pub elapsed_ms: f64,
}

/// Apply `policy` to the index. Returns a report describing what happened.
pub fn apply(ivf: &mut Ivf, policy: Policy, cfg: &PolicyConfig) -> PolicyReport {
    let label = match policy {
        Policy::None => "None",
        Policy::Split => "Split",
        Policy::Lazy => "Lazy",
        Policy::Hybrid => "Hybrid",
        Policy::FullRebuild => "FullRebuild",
    };
    let t = std::time::Instant::now();
    let mut splits = 0usize;
    let mut lazies = 0usize;
    match policy {
        Policy::None => {}
        Policy::Split => {
            splits = split_overloaded(ivf, cfg);
        }
        Policy::Lazy => {
            lazies = lazy_recenter(ivf, cfg);
        }
        Policy::Hybrid => {
            lazies = lazy_recenter(ivf, cfg);
            splits = split_overloaded(ivf, cfg);
        }
        Policy::FullRebuild => {
            full_rebuild(ivf, cfg);
        }
    }
    PolicyReport {
        policy: label,
        centroids_after: ivf.centroids.len(),
        splits_applied: splits,
        lazy_recenters_applied: lazies,
        elapsed_ms: t.elapsed().as_secs_f64() * 1000.0,
    }
}

/// Walk every list; for any whose population exceeds
/// `cfg.split_threshold * mean_load`, run a 2-means split and replace the
/// list and its centroid with the two resulting lists/centroids.
fn split_overloaded(ivf: &mut Ivf, cfg: &PolicyConfig) -> usize {
    if ivf.lists.is_empty() {
        return 0;
    }
    let total: usize = ivf.lists.iter().map(|l| l.len()).sum();
    let mean = (total as f32) / (ivf.lists.len() as f32);
    let ceiling = (cfg.split_threshold * mean).max(8.0) as usize;
    let n_lists = ivf.lists.len();
    let mut new_centroids: Vec<Vec<f32>> = Vec::with_capacity(n_lists);
    let mut new_lists: Vec<Vec<u32>> = Vec::with_capacity(n_lists);
    let mut splits = 0usize;
    let mut rng = SmallRng::new(cfg.seed);
    for ci in 0..n_lists {
        let list = std::mem::take(&mut ivf.lists[ci]);
        if list.len() <= ceiling {
            new_centroids.push(std::mem::take(&mut ivf.centroids[ci]));
            new_lists.push(list);
            continue;
        }
        splits += 1;
        let (c0, c1, l0, l1) = two_means(ivf, &list, cfg.split_kmeans_iters, &mut rng);
        new_centroids.push(c0);
        new_lists.push(l0);
        new_centroids.push(c1);
        new_lists.push(l1);
    }
    ivf.centroids = new_centroids;
    ivf.lists = new_lists;
    splits
}

/// Per-list 2-means. Returns (c0, c1, list0, list1).
fn two_means(
    ivf: &Ivf,
    list: &[u32],
    iters: usize,
    rng: &mut SmallRng,
) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>) {
    let dim = ivf.dim;
    // Seed centroids: two distinct members.
    let i0 = rng.usize(list.len());
    let mut i1 = rng.usize(list.len());
    if list.len() > 1 && i1 == i0 {
        i1 = (i0 + 1) % list.len();
    }
    let mut c0 = ivf.vector(list[i0]).to_vec();
    let mut c1 = ivf.vector(list[i1]).to_vec();
    let mut a0: Vec<u32> = Vec::new();
    let mut a1: Vec<u32> = Vec::new();
    for _ in 0..iters {
        a0.clear();
        a1.clear();
        for &id in list {
            let v = ivf.vector(id);
            if sq_l2(v, &c0) <= sq_l2(v, &c1) {
                a0.push(id);
            } else {
                a1.push(id);
            }
        }
        if !a0.is_empty() {
            c0 = mean_vec(ivf, &a0, dim);
        }
        if !a1.is_empty() {
            c1 = mean_vec(ivf, &a1, dim);
        }
    }
    (c0, c1, a0, a1)
}

fn mean_vec(ivf: &Ivf, ids: &[u32], dim: usize) -> Vec<f32> {
    let mut acc = vec![0.0f32; dim];
    for &id in ids {
        let v = ivf.vector(id);
        for d in 0..dim {
            acc[d] += v[d];
        }
    }
    let inv = 1.0 / (ids.len().max(1) as f32);
    for d in 0..dim {
        acc[d] *= inv;
    }
    acc
}

/// Compute per-list drift contributions, find the median, and recenter every
/// list whose contribution exceeds `lazy_threshold * median`.
fn lazy_recenter(ivf: &mut Ivf, cfg: &PolicyConfig) -> usize {
    let n_lists = ivf.lists.len();
    if n_lists == 0 {
        return 0;
    }
    let mut contribs: Vec<f32> = Vec::with_capacity(n_lists);
    for (c, list) in ivf.centroids.iter().zip(ivf.lists.iter()) {
        let s: f32 = list.iter().map(|&id| sq_l2(c, ivf.vector(id))).sum();
        contribs.push(s);
    }
    let mut sorted = contribs.clone();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2].max(1e-6);
    let cutoff = cfg.lazy_threshold * median;
    let mut applied = 0usize;
    let dim = ivf.dim;
    for ci in 0..n_lists {
        if contribs[ci] <= cutoff || ivf.lists[ci].is_empty() {
            continue;
        }
        let list = ivf.lists[ci].clone();
        ivf.centroids[ci] = mean_vec(ivf, &list, dim);
        applied += 1;
    }
    applied
}

/// Full rebuild: forget the lists, retrain centroids on a random sample of the
/// inserted vectors via k-means, then reassign every vector.
fn full_rebuild(ivf: &mut Ivf, cfg: &PolicyConfig) {
    let n_lists = ivf.lists.len();
    let n = ivf.n as usize;
    if n_lists == 0 || n == 0 {
        return;
    }
    let sample_size = (n.min(4096)).max(n_lists);
    let mut rng = SmallRng::new(cfg.seed.wrapping_add(0x1234));
    let mut sample: Vec<Vec<f32>> = Vec::with_capacity(sample_size);
    for _ in 0..sample_size {
        let id = rng.usize(n) as u32;
        sample.push(ivf.vector(id).to_vec());
    }
    let dim = ivf.dim;
    let mut centroids: Vec<Vec<f32>> = (0..n_lists)
        .map(|_| sample[rng.usize(sample.len())].clone())
        .collect();
    let mut assign = vec![0usize; sample.len()];
    for _ in 0..cfg.full_rebuild_iters {
        for (i, v) in sample.iter().enumerate() {
            assign[i] = nearest_centroid(v, &centroids);
        }
        let mut new = vec![vec![0.0f32; dim]; n_lists];
        let mut counts = vec![0u32; n_lists];
        for (i, v) in sample.iter().enumerate() {
            let c = assign[i];
            counts[c] += 1;
            for d in 0..dim {
                new[c][d] += v[d];
            }
        }
        for c in 0..n_lists {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    new[c][d] *= inv;
                }
                centroids[c] = std::mem::take(&mut new[c]);
            }
        }
    }
    ivf.centroids = centroids;
    ivf.lists = vec![Vec::new(); n_lists];
    for id in 0..ivf.n {
        let v = ivf.vector(id).to_vec();
        let c = nearest_centroid(&v, &ivf.centroids);
        ivf.lists[c].push(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SmallRng;

    fn build(dim: usize, n: usize, n_lists: usize, seed: u64) -> Ivf {
        let mut rng = SmallRng::new(seed);
        let pts: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..dim).map(|_| rng.normal()).collect())
            .collect();
        let mut ivf = Ivf::new(dim, n_lists);
        ivf.train(&pts, 6, seed);
        for v in &pts {
            ivf.add(v);
        }
        ivf
    }

    #[test]
    fn split_increases_list_count_when_overloaded() {
        // Build, then artificially overload one list to exercise the split path.
        let mut ivf = build(4, 600, 4, 11);
        let biggest = ivf
            .lists
            .iter()
            .enumerate()
            .max_by_key(|(_, l)| l.len())
            .map(|(i, _)| i)
            .unwrap();
        // Move every id from the other lists into the biggest list to push it
        // far above the mean. (Manually rewriting lists like this is only
        // valid in a test: the raw vectors are untouched.)
        let n_lists = ivf.n_lists();
        for ci in 0..n_lists {
            if ci != biggest {
                let drained: Vec<u32> = std::mem::take(&mut ivf.lists[ci]);
                ivf.lists[biggest].extend(drained);
            }
        }
        let cfg = PolicyConfig {
            split_threshold: 1.5,
            ..Default::default()
        };
        let before = ivf.n_lists();
        let r = apply(&mut ivf, Policy::Split, &cfg);
        assert!(r.splits_applied >= 1, "no splits: {r:?}");
        assert!(ivf.n_lists() > before);
    }

    #[test]
    fn lazy_lowers_drift_score() {
        let mut ivf = build(4, 500, 8, 17);
        // Shift only half the centroids — keeps the median low so the lazy
        // cutoff correctly identifies the drifted half as outliers.
        for (i, c) in ivf.centroids.iter_mut().enumerate() {
            if i % 2 == 0 {
                for d in c.iter_mut() {
                    *d += 1.5;
                }
            }
        }
        let before = ivf.drift_score();
        let r = apply(&mut ivf, Policy::Lazy, &PolicyConfig::default());
        let after = ivf.drift_score();
        assert!(r.lazy_recenters_applied > 0, "no recenters: {r:?}");
        assert!(after < before, "lazy must reduce drift: {before} -> {after}");
    }

    #[test]
    fn full_rebuild_preserves_count_and_list_total() {
        let mut ivf = build(4, 500, 8, 23);
        let n_before = ivf.n;
        apply(&mut ivf, Policy::FullRebuild, &PolicyConfig::default());
        let total: u32 = ivf.lists.iter().map(|l| l.len() as u32).sum();
        assert_eq!(total, n_before);
    }
}
