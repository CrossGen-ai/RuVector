//! Optimal B-bit codebook for the per-coordinate distribution of a
//! unit vector in D dimensions (approximately N(0, 1/D) for large D).
//!
//! Codebooks are stored in standard normal units (sigma=1); scaling by
//! 1/sqrt(D) happens at decode time. This lets one codebook serve all
//! dimensions and makes the Lloyd-Max fit a 1-D problem.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Codebook {
    pub bits: u8,
    /// Levels in standard-normal units. Length = 2^bits. Monotonically increasing.
    pub levels: Vec<f32>,
    /// Decision thresholds; levels.len() - 1 entries. threshold[i] separates
    /// level[i] from level[i+1].
    pub thresholds: Vec<f32>,
}

impl Codebook {
    /// Build a Lloyd-Max optimal codebook for the standard normal using
    /// deterministic numerical integration on a dense grid.
    pub fn gaussian(bits: u8) -> Self {
        assert!((1..=8).contains(&bits), "bits must be in 1..=8");
        let k = 1usize << bits;

        // Numerical grid over [-G, G] with step h.
        const G: f32 = 6.0;
        const N: usize = 20_001; // odd for symmetric centre
        let h = 2.0 * G / (N as f32 - 1.0);
        let xs: Vec<f32> = (0..N).map(|i| -G + h * (i as f32)).collect();
        let ps: Vec<f32> = xs.iter().map(|x| gauss_pdf(*x) * h).collect();

        // Initial quantile-based levels (bit-reversal order fine but equiprob is simpler).
        let mut levels: Vec<f32> = (0..k)
            .map(|i| gauss_inv_cdf(((i as f32) + 0.5) / (k as f32)))
            .collect();
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Lloyd-Max iterations.
        for _ in 0..100 {
            // Nearest-neighbour assignment on the grid.
            // thresholds[i] = (levels[i] + levels[i+1]) / 2
            let thr: Vec<f32> = (0..k - 1)
                .map(|i| 0.5 * (levels[i] + levels[i + 1]))
                .collect();
            let mut num = vec![0.0f64; k];
            let mut den = vec![0.0f64; k];
            for (x, p) in xs.iter().zip(&ps) {
                let mut bin = 0usize;
                while bin < thr.len() && *x > thr[bin] {
                    bin += 1;
                }
                num[bin] += *x as f64 * *p as f64;
                den[bin] += *p as f64;
            }
            let mut moved = 0.0f32;
            for i in 0..k {
                if den[i] > 1e-12 {
                    let new_l = (num[i] / den[i]) as f32;
                    moved = moved.max((new_l - levels[i]).abs());
                    levels[i] = new_l;
                }
            }
            if moved < 1e-5 {
                break;
            }
        }

        let thresholds: Vec<f32> = (0..k - 1)
            .map(|i| 0.5 * (levels[i] + levels[i + 1]))
            .collect();
        Self {
            bits,
            levels,
            thresholds,
        }
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.levels.len()
    }

    /// Theoretical MSE of this codebook on N(0,1) via grid.
    pub fn mse_gaussian(&self) -> f32 {
        const G: f32 = 6.0;
        const N: usize = 20_001;
        let h = 2.0 * G / (N as f32 - 1.0);
        let mut mse = 0.0f64;
        for i in 0..N {
            let x = -G + h * (i as f32);
            let p = gauss_pdf(x) * h;
            let q = self.encode_scalar(x);
            let d = self.levels[q as usize] - x;
            mse += d as f64 * d as f64 * p as f64;
        }
        mse as f32
    }

    /// Encode a single sample (in standard-normal units) to a code 0..k-1.
    #[inline]
    pub fn encode_scalar(&self, x: f32) -> u8 {
        // Linear scan is faster than binary search for k <= 256.
        let mut idx = 0u8;
        for t in &self.thresholds {
            if x > *t {
                idx += 1;
            } else {
                break;
            }
        }
        idx
    }

    /// Decode (in standard-normal units).
    #[inline]
    pub fn decode(&self, code: u8) -> f32 {
        self.levels[code as usize]
    }
}

#[inline]
fn gauss_pdf(x: f32) -> f32 {
    const INV_SQRT_2PI: f32 = 0.39894228;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Rational approximation of inverse standard normal CDF (Beasley-Springer).
/// Accurate enough to seed Lloyd-Max; final levels come from the iteration.
fn gauss_inv_cdf(p: f32) -> f32 {
    let a = [
        -3.969_683_028_665_376e1_f32,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b = [
        -5.447_609_879_822_406e1_f32,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c = [
        -7.784_894_002_430_293e-3_f32,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d = [
        7.784_695_709_041_462e-3_f32,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let p_low: f32 = 0.02425;
    let p_high: f32 = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_bit_matches_rabitq_sign() {
        let cb = Codebook::gaussian(1);
        assert_eq!(cb.k(), 2);
        // Levels should be roughly ±sqrt(2/pi) ≈ ±0.7979 (Lloyd-Max optimum).
        assert!((cb.levels[0] + 0.7979).abs() < 0.02);
        assert!((cb.levels[1] - 0.7979).abs() < 0.02);
    }

    #[test]
    fn mse_decreases_with_bits() {
        let mses: Vec<f32> = (1..=5).map(|b| Codebook::gaussian(b).mse_gaussian()).collect();
        for w in mses.windows(2) {
            assert!(w[0] > w[1], "MSE should strictly decrease: {:?}", mses);
        }
    }

    #[test]
    fn encode_decode_roundtrip_monotone() {
        let cb = Codebook::gaussian(4);
        let mut prev = f32::NEG_INFINITY;
        for x in [-3.0_f32, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0] {
            let decoded = cb.decode(cb.encode_scalar(x));
            assert!(decoded >= prev);
            prev = decoded;
        }
    }
}
