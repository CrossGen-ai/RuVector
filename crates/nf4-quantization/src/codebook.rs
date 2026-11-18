//! 16-level codebook with O(log k) encode and O(1) decode.

/// A sorted 16-level codebook (4-bit).
#[derive(Debug, Clone)]
pub struct Codebook16 {
    /// Sorted reconstruction levels in `[-1, 1]`.
    pub levels: [f32; 16],
}

impl Codebook16 {
    /// Construct from 16 levels. Panics if not strictly sorted.
    pub fn new(levels: [f32; 16]) -> Self {
        for w in levels.windows(2) {
            assert!(w[0] < w[1], "codebook must be strictly sorted");
        }
        Self { levels }
    }

    /// The canonical QLoRA NF4 codebook (16 quantiles of N(0,1), scaled to
    /// `[-1, 1]`). See QLoRA appendix E.2.
    pub fn qlora_nf4() -> Self {
        Self::new([
            -1.0,
            -0.6961928,
            -0.5250730,
            -0.3949414,
            -0.2844391,
            -0.1848577,
            -0.0915699,
            0.0,
            0.0795803,
            0.1609302,
            0.2461123,
            0.3379152,
            0.4407410,
            0.5626170,
            0.7229568,
            1.0,
        ])
    }

    /// Symmetric NF4 derived purely from CDF quantiles of N(0,1), rescaled
    /// to `[-1, 1]`. Useful as a self-contained baseline for signed data.
    pub fn symmetric_normal() -> Self {
        // 16 points at the (i+0.5)/16 quantiles of N(0,1), inverse via
        // Beasley-Springer-Moro (Acklam's approximation, |err|<1.15e-9).
        let mut l = [0f32; 16];
        for i in 0..16 {
            let p = (i as f64 + 0.5) / 16.0;
            l[i] = inv_phi(p) as f32;
        }
        // Normalize so the largest-abs entry is 1.0.
        let m = l.iter().fold(0f32, |a, &b| a.max(b.abs()));
        for v in &mut l {
            *v /= m;
        }
        Self::new(l)
    }

    /// Nearest-level encode via binary search on the sorted levels.
    #[inline]
    pub fn encode(&self, x: f32) -> u8 {
        // Clamp
        let x = x.clamp(self.levels[0], self.levels[15]);
        // Binary search for the insertion point.
        let mut lo = 0usize;
        let mut hi = 15usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.levels[mid] < x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        // lo is first level >= x. Compare to lo-1 for nearest.
        if lo == 0 {
            0
        } else {
            let a = self.levels[lo - 1];
            let b = self.levels[lo];
            if (x - a).abs() <= (b - x).abs() {
                (lo - 1) as u8
            } else {
                lo as u8
            }
        }
    }

    #[inline]
    pub fn decode(&self, code: u8) -> f32 {
        self.levels[(code & 0x0F) as usize]
    }

    /// Fit codebook levels to a sample via 1D Lloyd-Max iteration
    /// (k-means in R, k = 16). Returns the fitted codebook.
    pub fn fit_lloyd_max(sample: &[f32], iters: usize) -> Self {
        assert!(sample.len() >= 16, "need at least 16 sample points");
        // Initialize with uniform quantiles of the sample.
        let mut sorted: Vec<f32> = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut levels = [0f32; 16];
        for i in 0..16 {
            let idx = ((i as f64 + 0.5) / 16.0 * sorted.len() as f64) as usize;
            levels[i] = sorted[idx.min(sorted.len() - 1)];
        }
        // Lloyd iterations.
        let mut assign: Vec<u8> = vec![0; sample.len()];
        for _ in 0..iters {
            // Assign
            for (i, &x) in sample.iter().enumerate() {
                let mut best = 0u8;
                let mut bestd = f32::INFINITY;
                for (k, &c) in levels.iter().enumerate() {
                    let d = (x - c) * (x - c);
                    if d < bestd {
                        bestd = d;
                        best = k as u8;
                    }
                }
                assign[i] = best;
            }
            // Update
            let mut sum = [0f64; 16];
            let mut cnt = [0u32; 16];
            for (&x, &a) in sample.iter().zip(assign.iter()) {
                sum[a as usize] += x as f64;
                cnt[a as usize] += 1;
            }
            for k in 0..16 {
                if cnt[k] > 0 {
                    levels[k] = (sum[k] / cnt[k] as f64) as f32;
                }
            }
        }
        // Ensure strict ordering.
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Break any ties by small perturbation (rare).
        for i in 1..16 {
            if levels[i] <= levels[i - 1] {
                levels[i] = levels[i - 1] + 1e-6;
            }
        }
        // Normalize to [-1, 1].
        let m = levels.iter().fold(0f32, |a, &b| a.max(b.abs()));
        if m > 0.0 {
            for v in &mut levels {
                *v /= m;
            }
        }
        Self::new(levels)
    }
}

/// Acklam's inverse-normal approximation (double precision).
fn inv_phi(p: f64) -> f64 {
    // Coefficients
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qlora_sorted_and_signed() {
        let cb = Codebook16::qlora_nf4();
        assert_eq!(cb.levels[0], -1.0);
        assert_eq!(cb.levels[15], 1.0);
        assert!(cb.levels[7].abs() < 1e-6, "level 7 is the zero anchor");
    }

    #[test]
    fn round_trip_zero() {
        let cb = Codebook16::qlora_nf4();
        let c = cb.encode(0.0);
        assert_eq!(cb.decode(c), 0.0);
    }

    #[test]
    fn symmetric_encodes_extremes() {
        let cb = Codebook16::symmetric_normal();
        assert_eq!(cb.encode(10.0), 15);
        assert_eq!(cb.encode(-10.0), 0);
    }

    #[test]
    fn lloyd_max_learns_shifted_dist() {
        // Fit on a positively skewed distribution.
        let s: Vec<f32> = (0..4096).map(|i| (i as f32 / 4096.0).powf(2.0)).collect();
        let cb = Codebook16::fit_lloyd_max(&s, 20);
        // The top level should track the sample max (normalized to 1).
        assert!((cb.levels[15] - 1.0).abs() < 1e-5);
        // Levels strictly increasing.
        for w in cb.levels.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}
