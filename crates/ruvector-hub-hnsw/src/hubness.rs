//! Indegree statistics and Gini coefficient.
//!
//! "Hubness" is operationalised here as the indegree distribution of the
//! constructed graph. In a uniformly-mixing graph all nodes would receive
//! roughly `M` incoming edges; under hubness a long tail of nodes accumulates
//! many more, distorting greedy traversal.

#[derive(Debug, Clone, serde::Serialize)]
pub struct IndegreeStats {
    pub mean: f32,
    pub max: usize,
    pub p99: usize,
    pub gini: f32,
    /// Fraction of nodes whose indegree exceeds 3x the mean.
    pub hub_fraction: f32,
}

pub fn indegree_histogram(adj: &[Vec<u32>]) -> Vec<usize> {
    let n = adj.len();
    let mut indeg = vec![0usize; n];
    for edges in adj.iter() {
        for &nb in edges {
            indeg[nb as usize] += 1;
        }
    }
    indeg
}

pub fn gini(values: &[usize]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len() as f32;
    let sum: f32 = v.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    // Gini = (2 * sum_i(i * v_i) / (n * sum(v))) - (n + 1) / n  (1-indexed)
    let mut numerator = 0.0f32;
    for (i, &val) in v.iter().enumerate() {
        numerator += ((i as f32) + 1.0) * val;
    }
    (2.0 * numerator) / (n * sum) - (n + 1.0) / n
}

pub fn stats_from(adj: &[Vec<u32>]) -> IndegreeStats {
    let hist = indegree_histogram(adj);
    let n = hist.len().max(1) as f32;
    let sum: usize = hist.iter().sum();
    let mean = sum as f32 / n;
    let max = *hist.iter().max().unwrap_or(&0);
    let mut sorted = hist.clone();
    sorted.sort_unstable();
    let p99_idx = ((sorted.len() as f32) * 0.99) as usize;
    let p99 = *sorted.get(p99_idx.min(sorted.len() - 1)).unwrap_or(&0);
    let thresh = mean * 3.0;
    let hub_count = hist.iter().filter(|&&x| (x as f32) > thresh).count();
    IndegreeStats {
        mean,
        max,
        p99,
        gini: gini(&hist),
        hub_fraction: hub_count as f32 / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gini_uniform_is_zero() {
        let v = vec![5usize; 100];
        assert!(gini(&v).abs() < 1e-5);
    }
    #[test]
    fn gini_extreme_skew() {
        let mut v = vec![0usize; 100];
        v[0] = 100;
        let g = gini(&v);
        assert!(g > 0.9, "expected high gini for extreme skew, got {}", g);
    }
    #[test]
    fn indegree_counts() {
        let adj: Vec<Vec<u32>> = vec![vec![1, 2], vec![0], vec![0, 1]];
        let h = indegree_histogram(&adj);
        // node 0: from 1, 2 -> 2
        // node 1: from 0, 2 -> 2
        // node 2: from 0    -> 1
        assert_eq!(h, vec![2, 2, 1]);
    }
}
