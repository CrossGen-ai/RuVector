//! Shared retrieval metrics: `Hit` newtype and recall@k.

use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Recall@k: fraction of ground-truth top-k ids present in `results`.
pub fn recall_at_k(results: &[Hit], ground_truth: &[Hit], k: usize) -> f32 {
    let res_ids: HashSet<usize> = results.iter().take(k).map(|h| h.id).collect();
    let gt_ids: HashSet<usize> = ground_truth.iter().take(k).map(|h| h.id).collect();
    if gt_ids.is_empty() {
        return 1.0;
    }
    let inter = res_ids.intersection(&gt_ids).count();
    inter as f32 / k.min(gt_ids.len()) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect() {
        let gt = vec![
            Hit { id: 0, dist: 0.1 },
            Hit { id: 1, dist: 0.2 },
            Hit { id: 2, dist: 0.3 },
        ];
        assert!((recall_at_k(&gt, &gt, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_partial() {
        let gt = vec![
            Hit { id: 0, dist: 0.1 },
            Hit { id: 1, dist: 0.2 },
            Hit { id: 2, dist: 0.3 },
            Hit { id: 3, dist: 0.4 },
        ];
        let res = vec![
            Hit { id: 0, dist: 0.1 },
            Hit { id: 1, dist: 0.2 },
            Hit { id: 99, dist: 5.0 },
            Hit { id: 100, dist: 6.0 },
        ];
        assert!((recall_at_k(&res, &gt, 4) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recall_empty_gt_defaults_perfect() {
        let res = vec![Hit { id: 0, dist: 0.0 }];
        assert_eq!(recall_at_k(&res, &[], 5), 1.0);
    }
}
