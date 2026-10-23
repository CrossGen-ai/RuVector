//! Randomised Walsh–Hadamard rotation.
//!
//! The classical fast Walsh–Hadamard transform (FWHT) runs in O(d log d) and
//! is orthogonal (self-inverse up to scaling by 1/d). We precede it with a
//! random ±1 diagonal `D` to randomise its structure, giving the "randomised
//! Hadamard transform" (RHT) D·H. Multiplying a vector by D·H preserves L2
//! distances while making per-dimension variance highly uniform — exactly
//! what a uniform scalar quantiser wants.
//!
//! References:
//!   * Ailon & Chazelle 2006, "Approximate nearest neighbors and the
//!     fast Johnson–Lindenstrauss transform".
//!   * Yu et al 2016, "Orthogonal random features" (ICML/NeurIPS tradition).
//!   * Faiss `IndexPreTransform` / `RandomRotation`.

/// Deterministic ±1 pseudorandom sign vector from a 64-bit seed.
/// SplitMix64 → extract low bit of each output.
fn sign_vector(seed: u64, len: usize) -> Vec<i8> {
    let mut state: u64 = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            if z & 1 == 0 { 1 } else { -1 }
        })
        .collect()
}

/// In-place iterative Fast Walsh–Hadamard Transform.
/// After this, caller should scale by 1/sqrt(n) to keep norms (we do that in `Rotation::apply`).
fn fwht_inplace(v: &mut [f32]) {
    let n = v.len();
    debug_assert!(n.is_power_of_two(), "FWHT needs power-of-two length");
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
            i += h * 2;
        }
        h <<= 1;
    }
}

/// Randomised Walsh–Hadamard rotation. Pads to the next power of two.
#[derive(Clone, Debug)]
pub struct Rotation {
    original_dim: usize,
    padded_dim: usize,
    signs: Vec<i8>,
    norm: f32,
}

impl Rotation {
    /// Build a rotation for `dim`-dim vectors from a deterministic `seed`.
    pub fn new(dim: usize, seed: u64) -> Self {
        assert!(dim > 0, "dim must be positive");
        let padded_dim = dim.next_power_of_two();
        let signs = sign_vector(seed, padded_dim);
        let norm = 1.0 / (padded_dim as f32).sqrt();
        Self { original_dim: dim, padded_dim, signs, norm }
    }

    pub fn input_dim(&self) -> usize { self.original_dim }
    pub fn output_dim(&self) -> usize { self.padded_dim }

    /// Apply D·H to `x`, writing the (possibly padded) `padded_dim` output.
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        assert_eq!(x.len(), self.original_dim);
        assert_eq!(out.len(), self.padded_dim);
        for i in 0..self.original_dim { out[i] = x[i] * self.signs[i] as f32; }
        for v in out.iter_mut().skip(self.original_dim) { *v = 0.0; }
        fwht_inplace(out);
        for v in out.iter_mut() { *v *= self.norm; }
    }

    /// Rotate a whole dataset. Returns a flat Vec of length `n * padded_dim`.
    pub fn rotate_dataset(&self, data: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(data.len(), n * self.original_dim);
        let mut out = vec![0.0f32; n * self.padded_dim];
        for i in 0..n {
            let x = &data[i * self.original_dim..(i + 1) * self.original_dim];
            let y = &mut out[i * self.padded_dim..(i + 1) * self.padded_dim];
            self.apply(x, y);
        }
        out
    }
}

/// Alias: `Rotation` plays the role of a general D·H "Hadamard" transform.
pub type Hadamard = Rotation;

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
    }

    #[test]
    fn rotation_is_isometric_on_pairs() {
        // Pairwise L2 distances must be preserved up to rounding.
        let d = 48;
        let r = Rotation::new(d, 42);
        let n = 32;
        let mut data = vec![0.0f32; n * d];
        let mut state: u64 = 7;
        for v in data.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = (((state >> 33) as u32) as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        let rot = r.rotate_dataset(&data, n);
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &data[i * d..(i + 1) * d];
                let b = &data[j * d..(j + 1) * d];
                let ra = &rot[i * r.padded_dim..(i + 1) * r.padded_dim];
                let rb = &rot[j * r.padded_dim..(j + 1) * r.padded_dim];
                let d0 = l2(a, b);
                let d1 = l2(ra, rb);
                assert!((d0 - d1).abs() < 1e-3, "d0={} d1={}", d0, d1);
            }
        }
    }

    #[test]
    fn rotation_equalises_per_dim_variance() {
        // Skewed input: dim 0 has 10x variance of other dims. After rotation,
        // per-dim variance should flatten dramatically.
        let d = 64;
        let n = 2048;
        let mut data = vec![0.0f32; n * d];
        let mut state: u64 = 1;
        for i in 0..n {
            for j in 0..d {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((state >> 33) as u32) as f32 / u32::MAX as f32;
                let v = u * 2.0 - 1.0;
                data[i * d + j] = if j == 0 { v * 10.0 } else { v };
            }
        }
        let input_var = per_dim_variance(&data, n, d);
        let input_ratio = input_var.iter().cloned().fold(f32::MIN, f32::max)
            / input_var.iter().cloned().fold(f32::MAX, f32::min);
        let r = Rotation::new(d, 99);
        let rot = r.rotate_dataset(&data, n);
        let rot_var = per_dim_variance(&rot, n, r.padded_dim);
        let rot_ratio = rot_var.iter().cloned().fold(f32::MIN, f32::max)
            / rot_var.iter().cloned().fold(f32::MAX, f32::min);
        assert!(input_ratio > 50.0, "expected skewed input, got {}", input_ratio);
        assert!(rot_ratio < 5.0, "rotation should flatten variance, got {}", rot_ratio);
    }

    fn per_dim_variance(data: &[f32], n: usize, d: usize) -> Vec<f32> {
        let mut mean = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..d { mean[j] += data[i * d + j]; }
        }
        for m in &mut mean { *m /= n as f32; }
        let mut var = vec![0.0f32; d];
        for i in 0..n {
            for j in 0..d {
                let dv = data[i * d + j] - mean[j];
                var[j] += dv * dv;
            }
        }
        for v in &mut var { *v /= n as f32; }
        var
    }
}
