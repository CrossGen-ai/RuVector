//! In-place Fast Walsh-Hadamard Transform (unnormalized).
//! Length must be a power of two. Cost: O(n log n).

pub fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

pub fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two(), "fwht requires power-of-two length");
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
    // Normalize so that the transform preserves L2 norm.
    let s = 1.0f32 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// Apply a random-sign flip then FWHT. Equivalent to a structured
/// random orthogonal rotation (Ailon-Chazelle SRHT). Signs are derived
/// from `seed`.
pub fn srht(x: &mut [f32], seed: u64) {
    use rand::SeedableRng;
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    for v in x.iter_mut() {
        if rng.gen::<bool>() { *v = -*v; }
    }
    fwht(x);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_norm_preserved() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n0: f32 = v.iter().map(|x| x * x).sum();
        fwht(&mut v);
        let n1: f32 = v.iter().map(|x| x * x).sum();
        assert!((n0 - n1).abs() < 1e-4, "{} vs {}", n0, n1);
    }

    #[test]
    fn fwht_involution() {
        let mut v = vec![0.1f32, -0.4, 0.3, 0.9, -0.2, 0.5, 0.0, -0.7];
        let orig = v.clone();
        fwht(&mut v);
        fwht(&mut v);
        for (a, b) in v.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
