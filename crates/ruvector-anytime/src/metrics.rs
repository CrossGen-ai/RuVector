//! Benchmark metrics: recall, latency stats, memory estimation.

use std::time::Duration;

/// Recall@k between predicted neighbor ids and ground-truth neighbor ids.
pub fn recall_at_k(pred: &[u32], truth: &[u32], k: usize) -> f32 {
    if truth.is_empty() {
        return 0.0;
    }
    let k = k.min(truth.len());
    let truth_set: std::collections::HashSet<u32> = truth.iter().take(k).copied().collect();
    let hits = pred
        .iter()
        .take(k)
        .filter(|x| truth_set.contains(x))
        .count();
    hits as f32 / k as f32
}

/// Estimated bytes for vectors + adjacency.
pub fn memory_estimate_bytes(n: usize, dims: usize, degree: usize) -> usize {
    let vec_bytes = n * dims * std::mem::size_of::<f32>();
    let adj_bytes = n * degree * std::mem::size_of::<u32>();
    let overhead = n * std::mem::size_of::<Vec<u32>>();
    vec_bytes + adj_bytes + overhead
}

/// Simple p50/p95/p99 latency statistics from a vector of durations.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub mean_us: f64,
    pub n: usize,
}

impl LatencyStats {
    pub fn from_durations(mut samples: Vec<Duration>) -> Self {
        samples.sort_unstable();
        let n = samples.len();
        let pct = |p: f64| -> f64 {
            if n == 0 {
                return 0.0;
            }
            let idx = ((n as f64 - 1.0) * p).round() as usize;
            samples[idx.min(n - 1)].as_micros() as f64
        };
        let mean = if n == 0 {
            0.0
        } else {
            samples.iter().map(|d| d.as_micros() as f64).sum::<f64>() / n as f64
        };
        LatencyStats {
            p50_us: pct(0.50),
            p95_us: pct(0.95),
            p99_us: pct(0.99),
            mean_us: mean,
            n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect_when_all_match() {
        let pred = vec![1, 2, 3, 4, 5];
        let truth = vec![1, 2, 3, 4, 5];
        assert_eq!(recall_at_k(&pred, &truth, 5), 1.0);
    }

    #[test]
    fn recall_zero_on_disjoint() {
        let pred = vec![1, 2, 3];
        let truth = vec![10, 11, 12];
        assert_eq!(recall_at_k(&pred, &truth, 3), 0.0);
    }
}
