//! Discrete water-filling bit allocator.
//!
//! Given per-dimension variances `sigma2[i]`, an average bit budget `b_avg`,
//! and bounds `[b_lo, b_hi]`, we minimize
//!
//!   ∑ᵢ σᵢ² · 4^(-bᵢ)
//!
//! over integer `bᵢ ∈ [b_lo, b_hi]` subject to ∑bᵢ = round(D · b_avg).
//! The high-rate scalar-quantization distortion for a uniform quantizer over
//! a fixed range is proportional to 4^(-bᵢ); adding one bit to dim `i` reduces
//! its contribution by a factor of 4. So a greedy allocator that repeatedly
//! spends the marginal bit on the dim with the largest current contribution
//! is optimal (this is the discrete analog of the classic reverse-water-filling
//! result — see Cover & Thomas, "Elements of Information Theory", ch. 13).
//!
//! Complexity: O(D log D + B log D) with a heap; we use a Vec + linear scan
//! since D is typically ≤ 2048 in embedding workloads and the constant factor
//! wins for small D.

use crate::FasqError;

/// Compute integer bit counts `b[i]` per dimension.
///
/// * `sigma2` — per-dim variances (>= 0). Zero variances are handled: they
///   receive `b_lo` bits since spending more bits on them yields no distortion
///   reduction.
/// * `b_avg` — target average bits per dimension (fractional).
/// * `b_lo`, `b_hi` — inclusive bounds for each `b[i]`.
///
/// Returns a Vec<u8> of length `sigma2.len()`.
pub fn allocate(sigma2: &[f32], b_avg: f32, b_lo: u8, b_hi: u8) -> Result<Vec<u8>, FasqError> {
    if b_lo > b_hi || b_avg < b_lo as f32 || b_avg > b_hi as f32 {
        return Err(FasqError::InfeasibleBudget { avg: b_avg, min: b_lo, max: b_hi });
    }
    let d = sigma2.len();
    let total: i64 = (b_avg * d as f32).round() as i64;
    let total = total.clamp((b_lo as i64) * d as i64, (b_hi as i64) * d as i64);

    // Start every dim at b_lo, then greedily add bits.
    let mut bits: Vec<u8> = vec![b_lo; d];
    let mut spent: i64 = (b_lo as i64) * d as i64;

    // marginal[i] = current σ² · 4^(-b[i]) — reducing this by 1 bit gives
    // a distortion drop of 0.75 · marginal[i].
    let mut marginal: Vec<f32> = sigma2
        .iter()
        .map(|s| *s * 4f32.powi(-(b_lo as i32)))
        .collect();

    while spent < total {
        // Find argmax of marginal among dims not yet at b_hi.
        let mut best = usize::MAX;
        let mut best_val = f32::NEG_INFINITY;
        for i in 0..d {
            if bits[i] >= b_hi { continue; }
            // Skip strict zeros — they gain nothing from more bits.
            if sigma2[i] == 0.0 { continue; }
            if marginal[i] > best_val {
                best_val = marginal[i];
                best = i;
            }
        }
        if best == usize::MAX {
            // All non-zero dims saturated; give remaining bits to first
            // non-saturated (zero-variance) dim so we still hit the budget
            // — no distortion impact.
            for i in 0..d {
                if bits[i] < b_hi {
                    bits[i] += 1;
                    spent += 1;
                    break;
                }
            }
            // If truly nothing can grow, we stop early — feasible budget
            // is unreachable, caller was warned via InfeasibleBudget above
            // only in extreme cases.
            if spent < total { continue; } else { break; }
        }
        bits[best] += 1;
        marginal[best] *= 0.25; // one more bit ⇒ distortion drops by 4×
        spent += 1;
    }

    // Sanity: allocator must never exceed b_hi anywhere.
    debug_assert!(bits.iter().all(|b| *b >= b_lo && *b <= b_hi));
    Ok(bits)
}

/// Estimate total quantization distortion (unnormalized) for a given
/// allocation, useful for reporting.
pub fn expected_distortion(sigma2: &[f32], bits: &[u8]) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..sigma2.len() {
        acc += sigma2[i] as f64 * 4f64.powi(-(bits[i] as i32));
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_variance_gives_uniform_bits() {
        let sigma2 = vec![1.0f32; 8];
        let bits = allocate(&sigma2, 4.0, 2, 8).unwrap();
        assert!(bits.iter().all(|b| *b == 4));
    }

    #[test]
    fn high_variance_dim_gets_more_bits() {
        let sigma2 = vec![100.0, 1.0, 1.0, 1.0];
        let bits = allocate(&sigma2, 4.0, 2, 8).unwrap();
        // Total = 16, high-variance dim should be at ceiling.
        assert!(bits[0] >= bits[1]);
        assert!(bits[0] >= bits[2]);
        assert!(bits[0] >= bits[3]);
        assert_eq!(bits.iter().map(|b| *b as i32).sum::<i32>(), 16);
    }

    #[test]
    fn zero_variance_dim_receives_floor() {
        let sigma2 = vec![0.0, 100.0, 100.0, 100.0];
        let bits = allocate(&sigma2, 5.0, 2, 8).unwrap();
        assert_eq!(bits[0], 2);
        assert_eq!(bits.iter().map(|b| *b as i32).sum::<i32>(), 20);
    }

    #[test]
    fn respects_budget_edge_cases() {
        let sigma2 = vec![1.0f32; 4];
        let bits = allocate(&sigma2, 8.0, 2, 8).unwrap();
        assert!(bits.iter().all(|b| *b == 8));
        let bits = allocate(&sigma2, 2.0, 2, 8).unwrap();
        assert!(bits.iter().all(|b| *b == 2));
    }

    #[test]
    fn infeasible_budget_rejected() {
        let sigma2 = vec![1.0; 4];
        assert!(allocate(&sigma2, 1.5, 2, 8).is_err());
        assert!(allocate(&sigma2, 9.0, 2, 8).is_err());
    }
}
