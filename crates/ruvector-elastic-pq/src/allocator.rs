//! Bit-budget allocators.
//!
//! Every allocator returns a [`BitBudget`] — the vector of per-subspace
//! bit counts. The trainer then materializes a codebook of the requested
//! width for each subspace.

/// Strategy for distributing the total bit budget across subspaces.
#[derive(Clone, Debug)]
pub enum Allocator {
    /// Uniform bits per subspace (classical PQ).
    Uniform { bits: u8 },
    /// Bits allocated proportionally to each subspace's coordinate
    /// variance (a PCA-style prior). Total bits held constant.
    VarianceProportional {
        /// Total bit budget across all subspaces.
        total_bits: usize,
        /// Lower clamp per subspace.
        min_bits: u8,
        /// Upper clamp per subspace.
        max_bits: u8,
    },
    /// Start uniform, retrain, and iteratively swap one bit from the
    /// lowest-distortion-per-bit subspace to the highest-distortion-per-bit
    /// subspace as long as the exchange lowers total distortion.
    DistortionIterative {
        /// Initial uniform bits per subspace before swapping begins.
        start_bits: u8,
        /// Lower clamp per subspace.
        min_bits: u8,
        /// Upper clamp per subspace.
        max_bits: u8,
        /// Cap on the number of swap iterations.
        max_swaps: usize,
    },
}

/// Result of an allocator: `bits[m]` = number of bits for subspace `m`.
#[derive(Clone, Debug)]
pub struct BitBudget {
    /// Per-subspace bit count.
    pub bits: Vec<u8>,
}

impl BitBudget {
    /// Total bits per encoded vector across all subspaces.
    pub fn total_bits(&self) -> usize {
        self.bits.iter().map(|b| *b as usize).sum()
    }
}

/// Uniform allocator: give every subspace exactly `bits` bits.
pub fn uniform(m: usize, bits: u8) -> BitBudget {
    BitBudget {
        bits: vec![bits; m],
    }
}

/// Variance-proportional allocator. `sub_var[m]` is the total variance of
/// subspace `m` (sum of coordinate variances). The allocator picks integer
/// bit counts within `[min_bits, max_bits]` whose sum equals `total_bits`
/// and whose *fractional* target is proportional to `log2(sub_var[m])`.
///
/// We use log-variance because doubling variance calls for exactly one
/// extra bit (variance scales quadratically with the codebook radius
/// each bit buys).
pub fn variance_proportional(
    sub_var: &[f64],
    total_bits: usize,
    min_bits: u8,
    max_bits: u8,
) -> BitBudget {
    let m = sub_var.len();
    let min_total = m * min_bits as usize;
    let max_total = m * max_bits as usize;
    let mut target_total = total_bits.clamp(min_total, max_total);

    // Log-variance signal, floored so a zero-variance subspace still gets
    // the minimum.
    let raw: Vec<f64> = sub_var
        .iter()
        .map(|v| (v.max(1e-12).log2()).max(-40.0))
        .collect();
    let raw_min = raw.iter().cloned().fold(f64::INFINITY, f64::min);
    // Shift so raw' >= 0.
    let shifted: Vec<f64> = raw.iter().map(|r| r - raw_min).collect();
    let sum: f64 = shifted.iter().sum::<f64>().max(1.0);

    // Fractional bit target above the floor.
    let extra_budget = target_total.saturating_sub(min_total) as f64;
    let mut float_extra: Vec<f64> = shifted.iter().map(|s| s / sum * extra_budget).collect();

    // Integer floor, distribute remainders by largest fractional part.
    let mut bits: Vec<u8> = float_extra
        .iter()
        .map(|f| min_bits + f.floor() as u8)
        .collect();
    let mut given: usize = bits.iter().map(|b| *b as usize).sum();
    // Clamp to max_bits and roll leftover into others.
    for i in 0..m {
        if bits[i] > max_bits {
            let over = bits[i] - max_bits;
            bits[i] = max_bits;
            given = given.saturating_sub(over as usize);
            float_extra[i] = (max_bits - min_bits) as f64; // saturate
        }
    }
    // Distribute the remaining budget one bit at a time to the subspace
    // with the largest fractional remainder that isn't already at max.
    while given < target_total {
        let mut best = None;
        let mut best_frac = -1.0;
        for i in 0..m {
            if bits[i] >= max_bits {
                continue;
            }
            let frac = float_extra[i] - float_extra[i].floor();
            if frac > best_frac {
                best_frac = frac;
                best = Some(i);
            }
        }
        match best {
            Some(i) => {
                bits[i] += 1;
                float_extra[i] += 1.0;
                given += 1;
            }
            None => break, // everyone saturated
        }
    }
    // If we still under-shoot (all saturated), lower target so callers see
    // a truthful `total_bits()`.
    if given < target_total {
        target_total = given;
    }
    let _ = target_total;
    BitBudget { bits }
}

/// Elastic distortion-iterative allocator. The trainer will call this via
/// [`elastic_swap_step`] after every retraining pass.
pub fn distortion_iterative_seed(m: usize, start_bits: u8) -> BitBudget {
    uniform(m, start_bits)
}

/// One swap step. Given per-subspace distortions and the current budget,
/// find the (donor, receiver) pair that maximizes the estimated drop in
/// total distortion — donor is the subspace with the lowest marginal
/// distortion-per-bit (bits it can afford to lose), receiver the highest
/// marginal distortion-per-bit that can still accept a bit.
///
/// Returns `Some((donor, receiver))` if a beneficial swap exists,
/// `None` otherwise.
pub fn elastic_swap_step(
    distortion: &[f64],
    budget: &BitBudget,
    min_bits: u8,
    max_bits: u8,
) -> Option<(usize, usize)> {
    let m = budget.bits.len();
    // Marginal distortion drop from adding a bit: PQ theory says
    // distortion scales roughly like 2^(-2b/dim), so the fractional
    // savings from adding one bit is roughly (1 - 4^(-1/dim)) * current
    // distortion. Simpler surrogate that behaves the same on real data:
    // use distortion[i] / (bits[i]+1) as "expected drop if we add a bit"
    // and distortion[i] / bits[i] as "expected cost if we remove one".
    let mut best: Option<(usize, usize, f64)> = None;
    for r in 0..m {
        if budget.bits[r] >= max_bits {
            continue;
        }
        let gain_r = distortion[r] / (budget.bits[r] as f64 + 1.0);
        for d in 0..m {
            if d == r {
                continue;
            }
            if budget.bits[d] <= min_bits {
                continue;
            }
            let cost_d = distortion[d] / (budget.bits[d] as f64);
            let net = gain_r - cost_d;
            if net > 0.0 {
                if best.map_or(true, |(_, _, n)| net > n) {
                    best = Some((d, r, net));
                }
            }
        }
    }
    best.map(|(d, r, _)| (d, r))
}
