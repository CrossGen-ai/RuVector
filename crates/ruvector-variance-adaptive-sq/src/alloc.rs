//! Per-dimension bit allocation strategies.

#[derive(Clone, Copy, Debug)]
pub enum AllocPolicy {
    /// Every dimension gets the same number of bits.
    Uniform,
    /// Bits ∝ log2(variance + eps), scaled to meet the total budget,
    /// clipped to [b_min, b_max]. This is the VA-SQ policy.
    LogVariance,
    /// Bits ∝ variance (linear). Mostly a sanity ablation — tends to
    /// concentrate too many bits on a few high-variance dimensions.
    LinearVariance,
}

/// Allocate bits per dimension under a global budget.
///
/// `budget_bits_per_dim` is the *average* bits/dim; the total budget is
/// `budget_bits_per_dim * D`. Returned values are in [b_min, b_max] and
/// sum to the requested total (rounded, with any residual spread across
/// the highest-variance dims).
pub fn allocate_bits(
    variances: &[f32],
    budget_bits_per_dim: u32,
    b_min: u32,
    b_max: u32,
    policy: AllocPolicy,
) -> Vec<u32> {
    assert!(b_min <= budget_bits_per_dim && budget_bits_per_dim <= b_max);
    let d = variances.len();
    let total: i64 = (budget_bits_per_dim as i64) * (d as i64);

    if matches!(policy, AllocPolicy::Uniform) {
        return vec![budget_bits_per_dim; d];
    }

    let eps = 1e-6_f32;
    let scores: Vec<f32> = match policy {
        AllocPolicy::LogVariance => variances
            .iter()
            .map(|v| (v + eps).ln().max(0.0_f32))
            .collect(),
        AllocPolicy::LinearVariance => variances.iter().map(|v| v + eps).collect(),
        AllocPolicy::Uniform => unreachable!(),
    };
    let sum: f32 = scores.iter().sum();

    // Water-filling with clip: iteratively redistribute surplus from clipped
    // dims (at b_max) or deficit from clipped dims (at b_min) until stable.
    let mut bits: Vec<u32> = vec![0; d];
    let mut free: Vec<bool> = vec![true; d];
    let mut free_sum = sum;
    let mut remaining = total;

    for _ in 0..32 {
        let mut changed = false;
        for i in 0..d {
            if !free[i] {
                continue;
            }
            let share = if free_sum > 0.0 {
                (scores[i] / free_sum) * (remaining as f32)
            } else {
                remaining as f32 / (free.iter().filter(|f| **f).count().max(1) as f32)
            };
            let bi = share.round() as i64;
            if bi >= b_max as i64 {
                bits[i] = b_max;
                remaining -= b_max as i64;
                free_sum -= scores[i];
                free[i] = false;
                changed = true;
            } else if bi <= b_min as i64 {
                bits[i] = b_min;
                remaining -= b_min as i64;
                free_sum -= scores[i];
                free[i] = false;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for i in 0..d {
        if free[i] {
            let share = if free_sum > 0.0 {
                (scores[i] / free_sum) * (remaining as f32)
            } else {
                0.0
            };
            bits[i] = share.round().clamp(b_min as f32, b_max as f32) as u32;
        }
    }

    // Snap total to match the budget exactly by nudging the dims whose
    // fractional share is furthest from their rounded value.
    let mut have: i64 = bits.iter().map(|b| *b as i64).sum();
    let mut order: Vec<usize> = (0..d).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut idx = 0usize;
    while have < total && idx < d * 4 {
        let i = order[idx % d];
        if bits[i] < b_max {
            bits[i] += 1;
            have += 1;
        }
        idx += 1;
    }
    let mut idx = 0usize;
    let mut rev = order.clone();
    rev.reverse();
    while have > total && idx < d * 4 {
        let i = rev[idx % d];
        if bits[i] > b_min {
            bits[i] -= 1;
            have -= 1;
        }
        idx += 1;
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_matches_budget_exactly() {
        let v = vec![1.0_f32; 64];
        let b = allocate_bits(&v, 4, 1, 8, AllocPolicy::Uniform);
        assert_eq!(b.len(), 64);
        assert!(b.iter().all(|x| *x == 4));
    }

    #[test]
    fn log_variance_respects_total_budget() {
        let mut v = vec![0.01_f32; 64];
        for i in 0..8 {
            v[i] = 100.0;
        }
        let b = allocate_bits(&v, 4, 1, 8, AllocPolicy::LogVariance);
        let total: u32 = b.iter().sum();
        assert_eq!(total, 4 * 64);
        for i in 0..8 {
            assert!(b[i] >= 4, "high-variance dim {} got only {} bits", i, b[i]);
        }
    }

    #[test]
    fn clip_bounds_hold() {
        let v: Vec<f32> = (0..32).map(|i| i as f32 * 10.0).collect();
        let b = allocate_bits(&v, 3, 2, 5, AllocPolicy::LogVariance);
        for bi in &b {
            assert!(*bi >= 2 && *bi <= 5);
        }
    }
}
