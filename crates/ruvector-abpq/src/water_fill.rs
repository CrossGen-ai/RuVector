//! Water-filling bit allocation.
//!
//! Given per-subquantizer variances `v_i` (i = 0..m) and a total integer bit
//! budget B with per-subquantizer floor `bmin` and cap `bmax`, choose integer
//! bits `b_i` that minimize
//!
//! ```text
//! D(b) = sum_i v_i * 2^(-2 b_i)
//! ```
//!
//! subject to `sum_i b_i = B` and `bmin <= b_i <= bmax`.
//!
//! Continuous relaxation gives the classic reverse-water-filling solution
//! `b_i* = 0.5 * log2(v_i / lambda)`. For a strictly convex objective in the
//! continuous variables the integer optimum can be reached by a **greedy
//! swap**: start at all `bmin`, and repeatedly add one bit to whichever
//! subquantizer maximizes the marginal drop in D, until the budget is met.
//! Because doubling the codebook cuts distortion by ~4x, the marginal gain of
//! moving `b_i` to `b_i + 1` is exactly `0.75 * v_i * 2^(-2 b_i)`, which is
//! non-increasing in b_i — this makes the greedy step exact.
//!
//! We use an O(m*B) loop with a small priority-queue (m ≤ 32 in practice).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Compute per-subvector variance (trace of covariance) for a n*d matrix
/// split into m equal-size subvectors.
pub fn per_sub_variance(data: &[f32], d: usize, m: usize) -> Vec<f32> {
    assert!(d % m == 0);
    let n = data.len() / d;
    let dsub = d / m;
    let mut mean = vec![0f32; d];
    if n == 0 { return vec![0.0; m]; }
    for row in 0..n {
        let base = row * d;
        for j in 0..d { mean[j] += data[base + j]; }
    }
    let inv = 1.0 / n as f32;
    for j in 0..d { mean[j] *= inv; }
    let mut var = vec![0f32; d];
    for row in 0..n {
        let base = row * d;
        for j in 0..d {
            let diff = data[base + j] - mean[j];
            var[j] += diff * diff;
        }
    }
    for j in 0..d { var[j] *= inv; }
    let mut sub_var = vec![0f32; m];
    for i in 0..m {
        let mut s = 0f32;
        for j in 0..dsub { s += var[i * dsub + j]; }
        sub_var[i] = s;
    }
    sub_var
}

#[derive(Debug, Clone, Copy)]
struct Slot { gain: f32, idx: u32, bits: u8 }
impl Eq for Slot {}
impl PartialEq for Slot { fn eq(&self, other: &Self) -> bool { self.gain == other.gain } }
impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap on Ord
        self.gain.partial_cmp(&other.gain).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Slot { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }

/// Marginal distortion drop when bumping bits `b -> b+1` for variance v.
/// D(b) = v * 2^(-2b); drop = D(b) - D(b+1) = v*2^(-2b)*(1 - 1/4) = 0.75 v * 2^(-2b).
#[inline]
fn marginal_gain(v: f32, b: u8) -> f32 {
    let scale = 4f32.powi(-(b as i32));
    0.75 * v * scale
}

/// Allocate `total` bits across variances, clipped to [bmin, bmax] per slot.
/// Returns per-slot bit counts summing to `total`.
pub fn allocate_bits(
    variances: &[f32],
    total: usize,
    bmin: u8,
    bmax: u8,
) -> Result<Vec<u8>, String> {
    let m = variances.len();
    if bmin > bmax { return Err("bmin > bmax".into()); }
    if m == 0 { return Err("no subquantizers".into()); }
    let lo = bmin as usize * m;
    let hi = bmax as usize * m;
    if total < lo || total > hi {
        return Err(format!("total {} outside [{}, {}]", total, lo, hi));
    }
    let mut bits = vec![bmin; m];
    let mut remaining = total - lo;
    if remaining == 0 { return Ok(bits); }

    let mut heap: BinaryHeap<Slot> = BinaryHeap::with_capacity(m);
    for (i, &v) in variances.iter().enumerate() {
        if bmin < bmax {
            heap.push(Slot { gain: marginal_gain(v, bmin), idx: i as u32, bits: bmin });
        }
    }
    while remaining > 0 {
        let Some(top) = heap.pop() else { break };
        let i = top.idx as usize;
        // stale entry check
        if top.bits != bits[i] { continue; }
        bits[i] += 1;
        remaining -= 1;
        if bits[i] < bmax {
            heap.push(Slot { gain: marginal_gain(variances[i], bits[i]), idx: top.idx, bits: bits[i] });
        }
    }
    if remaining > 0 {
        return Err("could not allocate all bits (all slots at bmax)".into());
    }
    Ok(bits)
}

/// Predicted total distortion sum_i v_i * 2^(-2 b_i).
pub fn predicted_distortion(variances: &[f32], bits: &[u8]) -> f32 {
    variances.iter().zip(bits.iter())
        .map(|(v, &b)| v * 4f32.powi(-(b as i32)))
        .sum()
}
