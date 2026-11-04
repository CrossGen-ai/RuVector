//! Binary signature selector.
//!
//! Given `nlist` centroids in R^d, we:
//!   1. Zero-pad each centroid to the next power of two `p` and apply
//!      SRHT (sign-flip + Fast Walsh-Hadamard) with a fixed seed.
//!   2. Take the sign of each rotated coordinate, packing into
//!      `words = p / 64` u64 words. Memory per centroid = `words * 8`
//!      bytes, versus `4 * d` for the raw float vector.
//!   3. At query time, rotate the query the same way, binarize, and
//!      Hamming-score against every centroid signature using a simple
//!      XOR + `u64::count_ones` loop (maps to POPCNT on x86_64 and to
//!      CNT on aarch64 via LLVM autovectorization).
//!   4. Keep the top-M shortlist (M << nlist), then recompute exact L2
//!      only for those M centroids and return the top-N.

use crate::hadamard::{next_pow2, srht};
use crate::ProbeSelector;

pub struct BinarySignatureSelector {
    centroids: Vec<Vec<f32>>,
    sigs: Vec<Vec<u64>>,   // nlist x (words_per_rot * num_rots)
    mean: Vec<f32>,
    d: usize,
    p: usize,                  // padded power-of-two length per rotation
    words_per_rot: usize,
    num_rots: usize,
    shortlist_m: usize,
    rot_seeds: Vec<u64>,
}

impl BinarySignatureSelector {
    /// Convenience: a single-rotation selector (`num_rots = 1`).
    pub fn new(centroids: Vec<Vec<f32>>, shortlist_m: usize, rot_seed: u64) -> Self {
        Self::with_rotations(centroids, shortlist_m, rot_seed, 1)
    }

    /// `num_rots`: number of independent SRHT rotations to stack.
    /// Signature length = `num_rots * next_pow2(d)` bits. Memory per
    /// centroid grows linearly; Hamming compare cost grows linearly.
    /// Recall at a fixed shortlist improves roughly as sqrt(num_rots).
    pub fn with_rotations(
        centroids: Vec<Vec<f32>>,
        shortlist_m: usize,
        rot_seed: u64,
        num_rots: usize,
    ) -> Self {
        assert!(num_rots >= 1);
        let d = centroids[0].len();
        let p = next_pow2(d.max(64));
        let words_per_rot = p / 64;
        let total_words = words_per_rot * num_rots;

        let mut mean = vec![0.0f32; d];
        for c in &centroids {
            for i in 0..d { mean[i] += c[i]; }
        }
        let scale = 1.0f32 / centroids.len() as f32;
        for m in &mut mean { *m *= scale; }

        let rot_seeds: Vec<u64> =
            (0..num_rots).map(|k| rot_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(k as u64)).collect();

        let mut sigs = Vec::with_capacity(centroids.len());
        let mut buf = vec![0.0f32; p];
        for c in &centroids {
            let mut sig = Vec::with_capacity(total_words);
            for &seed in &rot_seeds {
                buf.iter_mut().for_each(|v| *v = 0.0);
                for i in 0..d { buf[i] = c[i] - mean[i]; }
                srht(&mut buf, seed);
                sig.extend(pack_signs(&buf, words_per_rot));
            }
            sigs.push(sig);
        }
        Self {
            centroids,
            sigs,
            mean,
            d,
            p,
            words_per_rot,
            num_rots,
            shortlist_m,
            rot_seeds,
        }
    }

    pub fn bytes_per_centroid(&self) -> usize { self.words_per_rot * self.num_rots * 8 }

    pub fn num_rotations(&self) -> usize { self.num_rots }

    fn query_sig(&self, query: &[f32]) -> Vec<u64> {
        let mut buf = vec![0.0f32; self.p];
        let mut sig = Vec::with_capacity(self.words_per_rot * self.num_rots);
        for &seed in &self.rot_seeds {
            buf.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..self.d { buf[i] = query[i] - self.mean[i]; }
            srht(&mut buf, seed);
            sig.extend(pack_signs(&buf, self.words_per_rot));
        }
        sig
    }

    #[inline]
    fn hamming(a: &[u64], b: &[u64]) -> u32 {
        let mut s: u32 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            s += (x ^ y).count_ones();
        }
        s
    }

    #[inline]
    fn l2(&self, q: &[f32], c: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..self.d {
            let dv = q[i] - c[i];
            s += dv * dv;
        }
        s
    }
}

impl ProbeSelector for BinarySignatureSelector {
    fn name(&self) -> &'static str { "binary-signature-shortlist" }

    fn select(&self, query: &[f32], n: usize) -> Vec<(u32, f32)> {
        let qsig = self.query_sig(query);

        // Pass 1: Hamming shortlist to M candidates.
        let m = self.shortlist_m.max(n).min(self.centroids.len());
        let mut hd: Vec<(u32, u32)> = self
            .sigs
            .iter()
            .enumerate()
            .map(|(i, s)| (i as u32, Self::hamming(&qsig, s)))
            .collect();
        hd.select_nth_unstable_by_key(m.saturating_sub(1).max(0), |x| x.1);
        hd.truncate(m);

        // Pass 2: exact L2 refine on the M survivors.
        let mut refined: Vec<(u32, f32)> = hd
            .iter()
            .map(|(i, _)| (*i, self.l2(query, &self.centroids[*i as usize])))
            .collect();
        let k = n.min(refined.len());
        refined.select_nth_unstable_by(k.saturating_sub(1).max(0), |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });
        refined.truncate(k);
        refined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        refined
    }
}

#[inline]
fn pack_signs(x: &[f32], words: usize) -> Vec<u64> {
    let mut out = vec![0u64; words];
    for (i, v) in x.iter().enumerate() {
        if *v >= 0.0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_roundtrip() {
        let v = vec![0.1f32, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
        let p = pack_signs(&v, 1);
        // bits 0,2,4,6 set
        assert_eq!(p[0] & 0xFF, 0b01010101);
    }

    #[test]
    fn hamming_zero_for_self() {
        let v = vec![1.0f32; 64];
        let p = pack_signs(&v, 1);
        assert_eq!(BinarySignatureSelector::hamming(&p, &p), 0);
    }
}
