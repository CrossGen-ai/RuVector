//! Radix-bucket top-K collector.
//!
//! Buckets are indexed by the biased f32 exponent of the candidate distance,
//! shifted so that the smallest bucket we care about sits at index 0. Every
//! push is O(1): compute the bucket index, append the candidate.
//!
//! The collector maintains a running _worst-admissible_ exponent: the index of
//! the bucket that currently contains the K-th smallest candidate. Buckets
//! whose index exceeds that value are dropped on access, so memory stays
//! bounded by the number of distinct magnitudes seen plus K.
//!
//! The ordering guarantee is "ascending by distance exponent, best-effort
//! within a bucket". For beam search this is sufficient because:
//!
//! * The visit order of candidates whose distances share the same binary
//!   exponent differs by less than a factor of 2, and beam search is
//!   indifferent to such ordering (it only prunes against the current K-th).
//! * `finalize` performs an in-bucket sort of the overflow bucket so the
//!   returned K candidates are exactly the K smallest distances the pusher
//!   observed, in ascending order. The output is identical to an exact
//!   top-K up to ties in the K-th place.

use crate::{cmp_cand, Candidate, TopKCollector};

/// Number of consecutive exponent buckets to keep in a ring. For f32 the
/// usable range of non-negative finite distances spans ~255 exponents, but
/// ANN distances live in a much narrower band (typical L2/cosine ratios
/// cover ~30 exponents end to end), so 64 is comfortable.
pub const BUCKETS: usize = 64;

/// Candidates whose distance is exactly zero (frequent for self-matches)
/// land in this bucket.
const ZERO_BUCKET: u8 = 0;

/// Radix-bucket top-K. Push is O(1); finalize is O(k + #buckets).
pub struct RadixBucketTopK {
    k: usize,
    /// One Vec per magnitude bucket. The index is `bucket_of(dist)`.
    buckets: [Vec<Candidate>; BUCKETS],
    /// Total number of candidates currently retained across buckets.
    len: usize,
    /// Index of the highest bucket that still contains an admissible
    /// candidate; buckets above this are known to be pruned.
    /// Starts at `BUCKETS as u8 - 1` (admit everything) and tightens.
    cutoff_bucket: u8,
    /// Running count of candidates in buckets <= cutoff_bucket. Used to
    /// tighten `cutoff_bucket` once we accumulate >= k candidates.
    running_count: usize,
}

impl RadixBucketTopK {
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        // Required by array-init trick below (array of Vec<Candidate>).
        const EMPTY: Vec<Candidate> = Vec::new();
        Self {
            k,
            buckets: [EMPTY; BUCKETS],
            len: 0,
            cutoff_bucket: (BUCKETS - 1) as u8,
            running_count: 0,
        }
    }

    /// Map a non-negative finite distance to a bucket index in [0, BUCKETS).
    ///
    /// We use the f32 unbiased exponent + 1 so that distances in [1, 2) go to
    /// bucket 1, [2, 4) to bucket 2, etc. A zero distance (unbiased exponent
    /// = -127) is pinned to bucket 0. Distances larger than 2^(BUCKETS-2) are
    /// clamped to the last bucket.
    #[inline]
    pub fn bucket_of(dist: f32) -> u8 {
        if dist <= 0.0 {
            return ZERO_BUCKET;
        }
        // f32 layout: sign(1) | exponent(8, bias 127) | mantissa(23)
        let bits = dist.to_bits();
        let biased_exp = ((bits >> 23) & 0xff) as i32;
        // biased_exp == 0 would be subnormal or zero; subnormals fall into
        // bucket 0 too (negligible for ANN).
        if biased_exp == 0 {
            return ZERO_BUCKET;
        }
        // Shift so dists in [1,2) land near the middle of the range.
        // unbiased = biased - 127. Bucket = unbiased + OFFSET.
        const OFFSET: i32 = 32;
        let b = (biased_exp - 127) + OFFSET;
        if b < 0 {
            ZERO_BUCKET
        } else if b >= BUCKETS as i32 {
            (BUCKETS - 1) as u8
        } else {
            b as u8
        }
    }

    /// Return the smallest exponent-bucket such that buckets 0..=b together
    /// contain >= k candidates. Used to tighten the admission cutoff.
    fn find_cutoff(&self) -> u8 {
        let mut running = 0usize;
        for (i, b) in self.buckets.iter().enumerate() {
            running += b.len();
            if running >= self.k {
                return i as u8;
            }
        }
        (BUCKETS - 1) as u8
    }

    /// Drop every bucket strictly above `cutoff` (including freeing its vec).
    fn prune_above(&mut self, cutoff: u8) {
        for i in (cutoff as usize + 1)..BUCKETS {
            if !self.buckets[i].is_empty() {
                self.len -= self.buckets[i].len();
                self.buckets[i].clear();
                self.buckets[i].shrink_to_fit();
            }
        }
    }
}

impl TopKCollector for RadixBucketTopK {
    #[inline]
    fn push(&mut self, cand: Candidate) {
        let b = Self::bucket_of(cand.dist);
        if b > self.cutoff_bucket {
            // Past the admission frontier; drop without touching memory.
            return;
        }
        self.buckets[b as usize].push(cand);
        self.len += 1;
        self.running_count += 1;

        // Once we've seen at least k candidates, we may be able to lower the
        // cutoff. We only pay for this check periodically (every k pushes) to
        // keep push O(1) amortized.
        if self.running_count >= self.k && self.running_count.is_power_of_two() {
            let new_cutoff = self.find_cutoff();
            if new_cutoff < self.cutoff_bucket {
                self.cutoff_bucket = new_cutoff;
                self.prune_above(new_cutoff);
            }
        }
    }

    fn kth_upper_bound(&self) -> f32 {
        // Upper bound = top of the current cutoff bucket. For bucket b this is
        // 2^(b - OFFSET + 1). We want a safe over-estimate so beam-search
        // pruning stays correct.
        if self.len < self.k {
            return f32::INFINITY;
        }
        let b = self.cutoff_bucket as i32;
        const OFFSET: i32 = 32;
        let unbiased = (b - OFFSET) + 1;
        if unbiased > 127 {
            f32::INFINITY
        } else if unbiased < -126 {
            f32::MIN_POSITIVE
        } else {
            let biased = (unbiased + 127) as u32;
            f32::from_bits(biased << 23)
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn finalize(mut self) -> Vec<Candidate> {
        // Collect buckets in ascending exponent order; stop once we have k.
        let mut out: Vec<Candidate> = Vec::with_capacity(self.k);
        for b in 0..BUCKETS {
            if out.len() >= self.k {
                break;
            }
            if self.buckets[b].is_empty() {
                continue;
            }
            // Sort just the buckets we touch so the output is total-ordered.
            self.buckets[b].sort_by(cmp_cand);
            let remaining = self.k - out.len();
            if self.buckets[b].len() <= remaining {
                out.extend(self.buckets[b].drain(..));
            } else {
                out.extend(self.buckets[b].drain(..remaining));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_monotone() {
        // Larger distance => bucket index non-decreasing.
        let points: [f32; 10] = [0.0, 1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1e4];
        let mut last = 0u8;
        for &d in &points {
            let b = RadixBucketTopK::bucket_of(d);
            assert!(b >= last, "bucket for {} went backwards: {} < {}", d, b, last);
            last = b;
        }
    }

    #[test]
    fn exact_small_input() {
        let mut c = RadixBucketTopK::new(3);
        let samples = [
            (0u32, 0.5f32),
            (1, 0.1),
            (2, 0.9),
            (3, 0.3),
            (4, 0.7),
            (5, 0.2),
            (6, 0.6),
            (7, 0.05),
            (8, 0.95),
            (9, 0.4),
        ];
        for (id, d) in samples {
            c.push(Candidate::new(id, d));
        }
        let got = c.finalize();
        assert_eq!(got.len(), 3);
        // The three smallest are 0.05, 0.10, 0.20
        assert!((got[0].dist - 0.05).abs() < 1e-6);
        assert!((got[1].dist - 0.10).abs() < 1e-6);
        assert!((got[2].dist - 0.20).abs() < 1e-6);
    }

    #[test]
    fn upper_bound_is_safe_over_estimate() {
        let mut c = RadixBucketTopK::new(5);
        for i in 0..1024 {
            // Distances in [0.5, 1.5) - all in the same binary exponent band.
            let d = 0.5 + (i as f32) * (1.0 / 1024.0);
            c.push(Candidate::new(i as u32, d));
        }
        let ub = c.kth_upper_bound();
        let finals = c.finalize();
        assert_eq!(finals.len(), 5);
        // Upper bound must be >= every retained distance.
        for f in &finals {
            assert!(ub >= f.dist, "ub {} < retained {}", ub, f.dist);
        }
    }
}
